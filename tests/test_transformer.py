"""Tests for the transformer block and the ViT encoder."""

import numpy as np
import pytest

from jepa.config import JEPAConfig
from jepa.masking import gather_tokens, sample_masks
from jepa.transformer import TransformerBlock, VisionTransformer


@pytest.fixture
def cfg():
    return JEPAConfig()


@pytest.fixture
def block():
    return TransformerBlock(embed_dim=128, num_heads=4)


@pytest.fixture
def encoder(cfg):
    return VisionTransformer(cfg.patch_size, cfg.embed_dim, cfg.depth,
                             cfg.num_heads, cfg.mlp_ratio, cfg.dropout)


def images(batch=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, 32, 32, 3)).astype("float32")


def tokens(batch=2, count=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, count, 128)).astype("float32")


def silence_the_branches(block):
    """Zero the output kernel of the attention and of the MLP."""
    block.attention.projection.kernel.assign(
        np.zeros(block.attention.projection.kernel.shape, dtype="float32"))

    last_dense = [l for l in block.mlp.layers if hasattr(l, "kernel")][-1]
    last_dense.kernel.assign(np.zeros(last_dense.kernel.shape, dtype="float32"))


# ---- the residual ----------------------------------------------------------

def test_a_block_with_silenced_branches_returns_its_input(block):
    x = tokens()
    block(x)
    silence_the_branches(block)

    np.testing.assert_allclose(np.array(block(x)), x, atol=1e-5)


def test_the_residual_path_is_not_normalised(block):
    """Pre-norm keeps the input scale."""
    x = 50.0 * tokens()

    out = np.array(block(x))

    assert out.std() > 10.0


@pytest.mark.parametrize("count", [1, 15, 64])
def test_a_block_keeps_the_shape_it_was_given(block, count):
    x = tokens(count=count)

    assert np.array(block(x)).shape == (2, count, 128)


# ---- the encoder -----------------------------------------------------------

def test_the_target_encoder_sees_every_patch(encoder, cfg):
    out = np.array(encoder(images()))

    assert out.shape == (4, cfg.num_patches, cfg.embed_dim)


def test_the_context_encoder_only_returns_what_it_was_shown(encoder, cfg):
    indices = np.array([0, 5, 9, 63])

    out = np.array(encoder(images(), indices=indices))

    assert out.shape == (4, len(indices), cfg.embed_dim)


def test_a_gathered_token_remembers_where_it_sat(encoder):
    """On a constant image, only the position distinguishes tokens."""
    flat = np.full((1, 32, 32, 3), 0.7, dtype="float32")

    top_left = np.array(encoder(flat, indices=np.array([0, 1])))
    lower_row = np.array(encoder(flat, indices=np.array([8, 9])))

    assert not np.allclose(top_left, lower_row, atol=1e-5)


@pytest.mark.parametrize("count", [1, 7, 33, 64])
def test_one_encoder_serves_every_draw(encoder, cfg, count):
    indices = np.arange(count)

    out = np.array(encoder(images(2), indices=indices))

    assert out.shape == (2, count, cfg.embed_dim)


def test_depth_means_distinct_blocks(encoder, cfg):
    assert len(encoder.blocks) == cfg.depth
    assert len({id(b) for b in encoder.blocks}) == cfg.depth


# ---- the two encoders together ---------------------------------------------

def test_a_real_draw_runs_through_both_encoders(cfg):
    rng = np.random.default_rng(cfg.seed)
    context_encoder = VisionTransformer(cfg.patch_size, cfg.embed_dim, cfg.depth,
                                        cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
    target_encoder = VisionTransformer(cfg.patch_size, cfg.embed_dim, cfg.depth,
                                       cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
    batch = images(8)
    context_indices, target_indices = sample_masks(cfg, rng)

    context = np.array(context_encoder(batch, indices=context_indices))
    full = target_encoder(batch)
    targets = [np.array(gather_tokens(full, t)) for t in target_indices]

    assert context.shape == (8, len(context_indices), cfg.embed_dim)
    assert len(targets) == cfg.num_target_blocks
    for target, indices in zip(targets, target_indices):
        assert target.shape == (8, len(indices), cfg.embed_dim)
