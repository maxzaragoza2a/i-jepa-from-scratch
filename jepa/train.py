"""Pre-train the JEPA on CIFAR-10.

One training step, for one batch:
    sample_masks          -> context_indices, target_indices (shared by the batch)
    model(...)            -> predictions, targets
    jepa_loss             -> scalar
    optimizer             -> context encoder and predictor updated
    update_target_encoder -> target encoder pulled towards the context encoder

Usage:
    python -m jepa.train                   # full run, config.epochs epochs
    python -m jepa.train --max-steps 20    # smoke test
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
import keras

from jepa.config import JEPAConfig
from jepa.data import build_pretrain_dataset
from jepa.masking import sample_masks
from jepa.model import JEPA, jepa_loss
from jepa.utils import (CosineWithWarmup, count_params, ema_momentum,
                        format_params, set_seed)


def build_optimizer(config: JEPAConfig, total_steps: int, warmup_steps: int):
    """AdamW with linear warmup then cosine decay of the learning rate.

    Args:
        config: learning_rate and weight_decay are read from it.
        total_steps: number of optimizer steps of the whole run.
        warmup_steps: number of steps of the linear warmup.

    Returns:
        keras.optimizers.AdamW.
    """
    schedule = CosineWithWarmup(config.learning_rate, total_steps, warmup_steps)
    optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=config.weight_decay)
    optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "mask_token"])
    return optimizer


def train_step(model: JEPA, optimizer, images, context_indices, target_indices):
    """One optimizer step on one batch. The EMA is not done here.

    Args:
        model: the JEPA, with init_target_encoder already called.
        optimizer: from build_optimizer.
        images: (B, H, W, C), standardised.
        context_indices: 1-D int array, the M visible grid positions.
        target_indices: list of 1-D int arrays, one per target block.

    Returns:
        Scalar loss tensor.
    """
    with tf.GradientTape() as tape:
        predictions, targets = model(images, context_indices, target_indices,
                                     training=True)
        loss = jepa_loss(predictions, targets)
    weights = model.trainable_weights
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    return loss


def train_one_epoch(model: JEPA, optimizer, dataset, rng, step: int,
                    total_steps: int, config: JEPAConfig, max_steps=None):
    """Run over the dataset once: mask, step, EMA, for every batch.

    Args:
        model, optimizer: as in train_step.
        dataset: from build_pretrain_dataset, yields (B, H, W, C) batches.
        rng: np.random.Generator used to draw the masks.
        step: global step at the start of the epoch.
        total_steps: global step count of the whole run, for the EMA schedule.
        config: masking and EMA hyperparameters.
        max_steps: stop once the global step reaches it, or None.

    Returns:
        (mean_loss, step): mean loss over the batches seen, and the global
        step at the end of the epoch.
    """
    losses = []
    for images in dataset:
        if max_steps is not None and step >= max_steps:
            break

        # One draw per batch, shared by every image of the batch.
        context_indices, target_indices = sample_masks(config, rng)
        loss = train_step(model, optimizer, images, context_indices, target_indices)

        # After the optimizer step: the target follows the updated context encoder.
        momentum = ema_momentum(step, total_steps, config.ema_start, config.ema_end)
        model.update_target_encoder(momentum)

        losses.append(float(loss))
        step += 1
        if step % 50 == 0:
            print(f"  step {step}/{total_steps}  loss {np.mean(losses[-50:]):.4f}",
                  end="\r", flush=True)

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return mean_loss, step


def save_checkpoint(model: JEPA, out_dir: str, name: str = "jepa"):
    """Save every weight of the model (both encoders and the predictor).

    Args:
        model: the JEPA.
        out_dir: folder, created if missing (ignored by git: checkpoints/).
        name: file name without extension.

    Returns:
        Path of the saved file.
    """
    # The whole JEPA: VisionTransformer is a Layer and has no save_weights.
    # To reload: JEPA(config), init_target_encoder(), then load_weights(path).
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.weights.h5")
    model.save_weights(path)
    return path


def train(config: JEPAConfig, epochs=None, max_steps=None,
          out_dir: str = "checkpoints"):
    """Full pre-training run.

    Args:
        config: every hyperparameter.
        epochs: overrides config.epochs when given.
        max_steps: stop after this many optimizer steps, for a smoke test.
        out_dir: where checkpoints are written.

    Returns:
        (model, history), history being a list of dicts, one per epoch:
        {"epoch", "loss", "lr", "momentum", "seconds"}.
    """
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    dataset = build_pretrain_dataset(config)
    steps_per_epoch = int(dataset.cardinality())

    epochs = epochs or config.epochs
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    # Capped at a tenth of the run so that a short run still reaches the
    # cosine decay. A full run (3 of 30 epochs) is not affected.
    warmup_steps = min(config.warmup_epochs * steps_per_epoch, total_steps // 10)

    # The optimizer comes last, once every trainable weight exists:
    # init_target_encoder builds the encoders, one call builds the predictor.
    model = JEPA(config)
    model.init_target_encoder()
    model(next(iter(dataset)), *sample_masks(config, rng))
    optimizer = build_optimizer(config, total_steps, warmup_steps)

    trainable, non_trainable = count_params(model)
    print(f"parameters: {format_params(trainable)} trainable, "
          f"{format_params(non_trainable)} frozen (target encoder, positions)")
    print(f"{epochs} epochs x {steps_per_epoch} steps, {total_steps} steps in all, "
          f"{warmup_steps} of warmup")

    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss, step = train_one_epoch(model, optimizer, dataset, rng, step,
                                     total_steps, config, max_steps)
        record = {
            "epoch": epoch,
            "loss": loss,
            "lr": float(optimizer.learning_rate),
            "momentum": ema_momentum(step, total_steps, config.ema_start, config.ema_end),
            "seconds": time.time() - t0,
        }
        history.append(record)
        print(f"epoch {epoch:3d}/{epochs}  loss {record['loss']:.4f}  "
              f"lr {record['lr']:.2e}  ema {record['momentum']:.4f}  "
              f"{record['seconds']:.0f}s")
        save_checkpoint(model, out_dir)

        if max_steps is not None and step >= max_steps:
            break

    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=None,
                        help="number of epochs (default: config.epochs)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many steps, for a smoke test")
    parser.add_argument("--out", default="checkpoints",
                        help="where to save the weights")
    args = parser.parse_args()

    train(JEPAConfig(), epochs=args.epochs, max_steps=args.max_steps,
          out_dir=args.out)


if __name__ == "__main__":
    main()
