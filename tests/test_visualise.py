"""Tests for the mask visualiser (pure array functions only)."""

import numpy as np
import pytest

from jepa.config import JEPAConfig
from jepa.data import standardise
from jepa.masking import block_to_indices, sample_masks
from jepa.visualise import (
    BLANK,
    DIM,
    block_bounds,
    context_view,
    denormalise,
    patch_region,
    target_view,
)


@pytest.fixture
def cfg():
    return JEPAConfig()


@pytest.fixture
def image(cfg):
    """Random image with no pixel equal to BLANK or zero."""
    rng = np.random.default_rng(0)
    shape = (cfg.image_size, cfg.image_size, cfg.num_channels)
    return rng.uniform(0.6, 1.0, shape).astype("float32")


# ---- geometry --------------------------------------------------------------

@pytest.mark.parametrize("index,expected_rows,expected_cols", [
    (0, (0, 4), (0, 4)),
    (7, (0, 4), (28, 32)),
    (8, (4, 8), (0, 4)),
    (10, (4, 8), (8, 12)),
    (63, (28, 32), (28, 32)),
])
def test_patch_region_maps_row_major(cfg, index, expected_rows, expected_cols):
    rows, cols = patch_region(index, cfg)

    assert (rows.start, rows.stop) == expected_rows
    assert (cols.start, cols.stop) == expected_cols


def test_block_bounds_recovers_the_rectangle(cfg):
    p = cfg.patch_size
    indices = block_to_indices(1, 2, 3, 4, cfg.grid_size)

    left, top, width, height = block_bounds(indices, cfg)

    assert (left, top) == (2 * p, 1 * p)
    assert (width, height) == (4 * p, 3 * p)


# ---- the views -------------------------------------------------------------

def test_context_view_hides_every_target_patch(cfg, image):
    context_indices, target_indices = sample_masks(cfg, np.random.default_rng(1))
    view = context_view(image, context_indices, cfg)

    for index in np.unique(np.concatenate(target_indices)):
        rows, cols = patch_region(index, cfg)
        assert np.all(view[rows, cols] == BLANK), (
            f"target patch {index} is still visible in the context view"
        )


def test_context_view_keeps_every_context_patch_untouched(cfg, image):
    context_indices, _ = sample_masks(cfg, np.random.default_rng(1))
    view = context_view(image, context_indices, cfg)

    for index in context_indices:
        rows, cols = patch_region(index, cfg)
        assert np.array_equal(view[rows, cols], image[rows, cols])


def test_context_view_blanks_everything_outside_the_context(cfg, image):
    context_indices = np.array([0, 1, 2, 9, 40])
    view = context_view(image, context_indices, cfg)

    for index in set(range(cfg.num_patches)) - set(context_indices.tolist()):
        rows, cols = patch_region(index, cfg)
        assert np.all(view[rows, cols] == BLANK), f"patch {index} should be hidden"


def test_target_view_highlights_targets_and_dims_the_rest(cfg, image):
    context_indices, target_indices = sample_masks(cfg, np.random.default_rng(1))
    view = target_view(image, target_indices, cfg)

    hidden = set(np.concatenate(target_indices).tolist())
    for index in hidden:
        rows, cols = patch_region(index, cfg)
        assert np.array_equal(view[rows, cols], image[rows, cols])

    for index in set(range(cfg.num_patches)) - hidden:
        rows, cols = patch_region(index, cfg)
        assert np.allclose(view[rows, cols], image[rows, cols] * DIM)


# ---- denormalise -----------------------------------------------------------

def test_denormalise_inverts_standardise(cfg):
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)

    restored = denormalise(standardise(raw, cfg), cfg)

    assert np.allclose(restored, raw / 255.0, atol=1e-5)


def test_denormalise_clips_into_the_displayable_range(cfg):
    extreme = np.array([[[[-9.0, 0.0, 9.0]]]], dtype="float32")

    out = denormalise(extreme, cfg)

    assert out.min() >= 0.0 and out.max() <= 1.0
