"""Reproducibility, parameter counts and schedules."""

import os
import random

import numpy as np
import keras
from keras import ops


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and the Keras backend."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def count_params(model: keras.Model) -> tuple:
    """Return (trainable, non_trainable) parameter counts for a built model."""
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return trainable, non_trainable


def format_params(n: int) -> str:
    """4823296 -> '4.82M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class CosineWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, base_lr, total_steps, warmup_steps, final_lr_frac=0.0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.final_lr_frac = float(final_lr_frac)

    def __call__(self, step):
        step = ops.cast(step, "float32")
        warmup = max(float(self.warmup_steps), 1.0)

        warm_lr = self.base_lr * step / warmup

        decay_steps = max(float(self.total_steps - self.warmup_steps), 1.0)
        progress = ops.clip((step - warmup) / decay_steps, 0.0, 1.0)
        cosine = 0.5 * (1.0 + ops.cos(np.pi * progress))
        cos_lr = self.base_lr * (self.final_lr_frac + (1 - self.final_lr_frac) * cosine)

        return ops.where(step < warmup, warm_lr, cos_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "final_lr_frac": self.final_lr_frac,
        }


def ema_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    """EMA momentum of the target encoder, cosine-annealed from start to end."""
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return end - (end - start) * (1.0 + np.cos(np.pi * progress)) / 2.0
