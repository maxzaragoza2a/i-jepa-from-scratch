"""Round-trip tests for the patch pipeline."""

import numpy as np
import pytest

from jepa.config import JEPAConfig
from jepa.patch_embed import (
    PatchEmbedding,
    extract_patches,
    get_2d_sincos_pos_embed,
)


def unpatchify(patches, patch_size, grid_h, grid_w, channels):
    """Inverse of extract_patches: (B, N, p*p*C) -> (B, H, W, C)."""
    b = patches.shape[0]
    x = patches.reshape(b, grid_h, grid_w, patch_size, patch_size, channels)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, grid_h * patch_size, grid_w * patch_size, channels)


# (batch, height, width, channels, patch_size)
SHAPES = [
    (2, 32, 32, 3, 4),
    (1, 8, 8, 1, 2),
    (3, 16, 24, 3, 8),
]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("b,h,w,c,p", SHAPES)
def test_patching_is_exactly_reversible(rng, b, h, w, c, p):
    images = rng.standard_normal((b, h, w, c)).astype("float32")

    patches = np.asarray(extract_patches(images, p))
    restored = unpatchify(patches, p, h // p, w // p, c)

    assert np.array_equal(restored, images)


@pytest.mark.parametrize("b,h,w,c,p", SHAPES)
def test_patching_neither_creates_nor_destroys_values(rng, b, h, w, c, p):
    images = rng.standard_normal((b, h, w, c)).astype("float32")

    patches = np.asarray(extract_patches(images, p))

    assert patches.size == images.size
    assert np.array_equal(np.sort(patches, axis=None), np.sort(images, axis=None))


@pytest.mark.parametrize("b,h,w,c,p", SHAPES)
def test_patch_order_is_row_major(rng, b, h, w, c, p):
    images = rng.standard_normal((b, h, w, c)).astype("float32")
    grid_h, grid_w = h // p, w // p

    patches = np.asarray(extract_patches(images, p))

    for i in range(grid_h * grid_w):
        r, col = divmod(i, grid_w)
        block = images[:, r * p:(r + 1) * p, col * p:(col + 1) * p, :]
        assert np.array_equal(patches[:, i, :], block.reshape(b, -1))


def test_embedding_round_trip_recovers_the_image(rng):
    """Subtract positions, invert the Dense by least squares, unpatchify."""
    cfg = JEPAConfig()
    p, c = cfg.patch_size, cfg.num_channels
    patch_dim = p * p * c

    # The projection is only invertible while it widens.
    assert cfg.embed_dim >= patch_dim, (
        f"embed_dim {cfg.embed_dim} < patch_dim {patch_dim}: the projection "
        f"compresses and the image is genuinely unrecoverable"
    )

    images = rng.standard_normal(
        (2, cfg.image_size, cfg.image_size, c)
    ).astype("float32")

    layer = PatchEmbedding(p, cfg.embed_dim)
    tokens = np.asarray(layer(images))

    kernel = np.asarray(layer.projection.kernel)   # (patch_dim, embed_dim)
    bias = np.asarray(layer.projection.bias)
    pos = np.asarray(layer.pos_embed)

    assert np.linalg.matrix_rank(kernel) == patch_dim

    patches = (tokens - bias - pos) @ np.linalg.pinv(kernel)
    restored = unpatchify(patches, p, cfg.grid_size, cfg.grid_size, c)

    assert np.allclose(restored, images, atol=1e-4)


def test_positions_are_what_make_identical_patches_differ(rng):
    cfg = JEPAConfig()
    p = cfg.patch_size
    layer = PatchEmbedding(p, cfg.embed_dim)
    layer.build((None, cfg.image_size, cfg.image_size, cfg.num_channels))

    first = np.zeros((1, cfg.image_size, cfg.image_size, cfg.num_channels), "float32")
    second = np.zeros_like(first)
    first[0, 0:p, 0:p, :] = 1.0            # grid cell (0, 0) -> token 0
    second[0, p:2 * p, p:2 * p, :] = 1.0   # grid cell (1, 1) -> token 9

    token_a = np.asarray(layer(first))[0, 0]
    token_b = np.asarray(layer(second))[0, 9]
    assert not np.allclose(token_a, token_b)

    pos = np.asarray(layer.pos_embed)
    assert np.allclose(token_a - pos[0], token_b - pos[9], atol=1e-5)


# ---- input guards ----------------------------------------------------------

@pytest.mark.parametrize("h,w", [(30, 32), (32, 30), (33, 33)])
def test_extract_patches_rejects_a_size_it_cannot_tile(h, w):
    images = np.zeros((1, h, w, 3), "float32")

    with pytest.raises(ValueError, match="not divisible by patch_size"):
        extract_patches(images, 4)


@pytest.mark.parametrize("embed_dim", [2, 6, 126, 130])
def test_sincos_rejects_a_width_it_would_silently_narrow(embed_dim):
    with pytest.raises(ValueError, match="not divisible by 4"):
        get_2d_sincos_pos_embed(embed_dim, 8)


@pytest.mark.parametrize("embed_dim", [4, 64, 128])
def test_sincos_returns_the_full_requested_width(embed_dim):
    assert get_2d_sincos_pos_embed(embed_dim, 8).shape == (64, embed_dim)
