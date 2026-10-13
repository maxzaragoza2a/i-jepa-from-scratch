"""Tests for the assembled JEPA model."""

import numpy as np
import pytest
import tensorflow as tf

from jepa.config import JEPAConfig
from jepa.masking import sample_masks
from jepa.model import JEPA, jepa_loss


@pytest.fixture
def cfg():
    return JEPAConfig()


@pytest.fixture
def model(cfg):
    model = JEPA(cfg)
    model.init_target_encoder()
    return model


def images(batch=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, 32, 32, 3)).astype("float32")


def draw(cfg, seed=0):
    return sample_masks(cfg, np.random.default_rng(seed))


def as_arrays(weights):
    return [np.array(w) for w in weights]


# ---- the loss --------------------------------------------------------------

def test_the_loss_is_zero_when_predictions_match():
    x = [np.ones((2, 4, 8), "float32"), np.zeros((2, 6, 8), "float32")]

    assert float(jepa_loss(x, x)) == 0.0


def test_the_loss_averages_the_blocks_not_the_tokens():
    """A small block weighs as much as a big one."""
    small = np.zeros((1, 1, 1), "float32")
    big = np.zeros((1, 9, 1), "float32")

    loss = jepa_loss([small, big], [small + 2.0, big])

    np.testing.assert_allclose(float(loss), (4.0 + 0.0) / 2, rtol=1e-6)


# ---- the three parts -------------------------------------------------------

def test_the_predictor_is_the_narrow_one(model, cfg):
    assert len(model.predictor.blocks) == cfg.predictor_depth
    assert model.predictor.num_heads == cfg.predictor_num_heads
    assert model.predictor.predictor_embed_dim == cfg.predictor_embed_dim


def test_the_target_encoder_starts_as_a_copy_of_the_context_encoder(model):
    context = model.context_encoder.weights
    target = model.target_encoder.weights

    assert len(context) > 0
    assert [w.shape for w in context] == [w.shape for w in target]
    for c, t in zip(as_arrays(context), as_arrays(target)):
        np.testing.assert_array_equal(c, t)


def test_the_target_encoder_is_left_to_the_ema(model, cfg):
    x = images()
    context_indices, target_indices = draw(cfg)
    model(x, context_indices, target_indices)

    trainable = {id(w) for w in model.trainable_weights}

    assert model.target_encoder.trainable_weights == []
    assert not any(id(w) in trainable for w in model.target_encoder.weights)
    assert all(id(w) in trainable for w in model.context_encoder.trainable_weights)
    assert all(id(w) in trainable for w in model.predictor.trainable_weights)


# ---- the EMA ---------------------------------------------------------------

def perturb_context_encoder(model):
    for w in model.context_encoder.trainable_weights:
        w.assign(w + 1.0)


def test_a_momentum_of_one_freezes_the_target(model):
    before = as_arrays(model.target_encoder.weights)
    perturb_context_encoder(model)

    model.update_target_encoder(1.0)

    for b, a in zip(before, as_arrays(model.target_encoder.weights)):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_a_momentum_of_zero_copies_the_context(model):
    perturb_context_encoder(model)

    model.update_target_encoder(0.0)

    for c, t in zip(as_arrays(model.context_encoder.weights),
                    as_arrays(model.target_encoder.weights)):
        np.testing.assert_allclose(t, c, atol=1e-6)


def test_the_ema_moves_the_target_part_of_the_way(model):
    target = as_arrays(model.target_encoder.weights)
    perturb_context_encoder(model)
    context = as_arrays(model.context_encoder.weights)

    model.update_target_encoder(0.9)

    for t, c, new in zip(target, context, as_arrays(model.target_encoder.weights)):
        np.testing.assert_allclose(new, 0.9 * t + 0.1 * c, atol=1e-5)


# ---- the forward pass ------------------------------------------------------

def test_one_prediction_and_one_target_per_block(model, cfg):
    context_indices, target_indices = draw(cfg)

    predictions, targets = model(images(), context_indices, target_indices)

    assert len(predictions) == len(targets) == cfg.num_target_blocks
    for p, t, indices in zip(predictions, targets, target_indices):
        assert p.shape == t.shape == (2, len(indices), cfg.embed_dim)


def test_the_targets_are_the_full_image_encoding_gathered(model, cfg):
    """The target encoder sees every patch, not only the target block."""
    x = images()
    _, target_indices = draw(cfg)

    full = np.array(model.target_encoder(x))
    targets = model.encode_targets(x, target_indices)

    for t, indices in zip(targets, target_indices):
        np.testing.assert_allclose(np.array(t), full[:, indices], atol=1e-5)


def test_gradients_reach_the_context_encoder_and_the_predictor_only(model, cfg):
    x = images()
    context_indices, target_indices = draw(cfg)

    with tf.GradientTape(persistent=True) as tape:
        # Non-trainable variables are not watched by default.
        tape.watch([w.value for w in model.target_encoder.weights])
        predictions, targets = model(x, context_indices, target_indices, training=True)
        loss = jepa_loss(predictions, targets)
        target_sum = sum(tf.reduce_sum(t) for t in targets)

    grads = tape.gradient(loss, model.trainable_weights)
    assert all(g is not None for g in grads)
    assert float(loss) > 0.0

    # Watched or not, stop_gradient cuts the target branch.
    target_grads = tape.gradient(target_sum, [w.value for w in model.target_encoder.weights])
    assert all(g is None for g in target_grads)
