"""Tests for the predictor."""

import numpy as np
import pytest

from jepa.config import JEPAConfig
from jepa.masking import sample_masks
from jepa.patch_embed import get_2d_sincos_pos_embed
from jepa.predictor import Predictor
from jepa.transformer import VisionTransformer


@pytest.fixture
def cfg():
    return JEPAConfig()


@pytest.fixture
def predictor(cfg):
    return Predictor(cfg.grid_size, cfg.embed_dim, cfg.predictor_embed_dim,
                     cfg.predictor_depth, cfg.predictor_num_heads,
                     cfg.mlp_ratio, cfg.dropout)


def context(batch=2, count=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, count, 128)).astype("float32")


CONTEXT_INDICES = np.arange(20)
TARGET_INDICES = np.array([36, 37, 44, 45])


# ---- shapes ----------------------------------------------------------------

def test_one_prediction_per_target_at_the_encoder_width(predictor, cfg):
    out = np.array(predictor(context(), CONTEXT_INDICES, TARGET_INDICES))

    assert out.shape == (2, len(TARGET_INDICES), cfg.embed_dim)


@pytest.mark.parametrize("count", [1, 6, 12, 40])
def test_one_predictor_serves_every_target_size(predictor, cfg, count):
    target_indices = np.arange(20, 20 + count)

    out = np.array(predictor(context(), CONTEXT_INDICES, target_indices))

    assert out.shape == (2, count, cfg.embed_dim)


def test_the_predictor_is_narrower_than_the_encoder(predictor, cfg):
    assert len(predictor.blocks) == cfg.predictor_depth
    assert all(b.embed_dim == cfg.predictor_embed_dim for b in predictor.blocks)
    assert cfg.predictor_embed_dim < cfg.embed_dim


# ---- the weights -----------------------------------------------------------

def test_the_mask_token_is_learned_and_the_positions_are_not(predictor):
    predictor(context(), CONTEXT_INDICES, TARGET_INDICES)

    trainable = {id(w) for w in predictor.trainable_weights}

    assert id(predictor.mask_token) in trainable
    assert id(predictor.pos_embed) not in trainable


def test_the_positions_are_the_sincos_code_at_the_predictor_width(predictor, cfg):
    predictor(context(), CONTEXT_INDICES, TARGET_INDICES)

    expected = get_2d_sincos_pos_embed(cfg.predictor_embed_dim, cfg.grid_size)

    assert predictor.pos_embed.shape == (cfg.num_patches, cfg.predictor_embed_dim)
    np.testing.assert_allclose(np.array(predictor.pos_embed), expected, atol=1e-6)


# ---- what a prediction depends on ------------------------------------------

def test_a_prediction_depends_on_where_the_target_sits(predictor):
    """Mask tokens share one vector, only the position tells them apart."""
    x = context()

    here = np.array(predictor(x, CONTEXT_INDICES, np.array([36, 37])))
    there = np.array(predictor(x, CONTEXT_INDICES, np.array([62, 63])))

    assert not np.allclose(here, there, atol=1e-5)
    assert not np.allclose(here[:, 0], here[:, 1], atol=1e-5)


def test_a_prediction_depends_on_where_the_context_sat(predictor):
    x = context()

    top = np.array(predictor(x, np.arange(20), TARGET_INDICES))
    shifted = np.array(predictor(x, np.arange(8, 28), TARGET_INDICES))

    assert not np.allclose(top, shifted, atol=1e-5)


def test_a_prediction_depends_on_what_the_context_holds(predictor):
    one = np.array(predictor(context(seed=0), CONTEXT_INDICES, TARGET_INDICES))
    other = np.array(predictor(context(seed=1), CONTEXT_INDICES, TARGET_INDICES))

    assert not np.allclose(one, other, atol=1e-5)


def test_images_of_a_batch_do_not_see_each_other(predictor):
    x = context(batch=3)

    together = np.array(predictor(x, CONTEXT_INDICES, TARGET_INDICES))
    alone = np.array(predictor(x[:1], CONTEXT_INDICES, TARGET_INDICES))

    np.testing.assert_allclose(together[:1], alone, atol=1e-5)


# ---- config ----------------------------------------------------------------

def test_a_predictor_can_be_rebuilt_from_its_config(predictor):
    clone = Predictor.from_config(predictor.get_config())

    out = np.array(clone(context(), CONTEXT_INDICES, TARGET_INDICES))

    assert clone.get_config() == predictor.get_config()
    assert out.shape == (2, len(TARGET_INDICES), predictor.embed_dim)


# ---- encoder and predictor together ----------------------------------------

def test_a_real_draw_runs_through_the_encoder_and_the_predictor(predictor, cfg):
    rng = np.random.default_rng(cfg.seed)
    encoder = VisionTransformer(cfg.patch_size, cfg.embed_dim, cfg.depth,
                                cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
    batch = rng.standard_normal((8, 32, 32, 3)).astype("float32")
    context_indices, target_indices = sample_masks(cfg, rng)

    context_tokens = encoder(batch, indices=context_indices)
    predictions = [np.array(predictor(context_tokens, context_indices, t))
                   for t in target_indices]

    assert len(predictions) == cfg.num_target_blocks
    for prediction, indices in zip(predictions, target_indices):
        assert prediction.shape == (8, len(indices), cfg.embed_dim)
