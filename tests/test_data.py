"""Tests for the input pipeline, on CIFAR-10 and STL-10.

Statistics are accumulated in float64: a float32 accumulator is not precise
enough over the full train split. These tests read the real datasets.
"""

import numpy as np
import pytest
import tensorflow as tf

from jepa.config import DATASETS, JEPAConfig
from jepa.data import (
    augment,
    build_pretrain_dataset,
    build_probe_dataset,
    load_split,
    standardise,
)


@pytest.fixture(scope="module", params=sorted(DATASETS))
def cfg(request):
    return JEPAConfig(**DATASETS[request.param])


@pytest.fixture(scope="module")
def first_image(cfg):
    """First uint8 image of the pretraining split, as the pipeline sees it."""
    image, _ = next(iter(load_split(cfg, "train", pretrain=True)))
    return image


@pytest.fixture(scope="module")
def pixel_stats(cfg):
    """Per-channel mean and std of the pretraining split, before and after
    standardise, in one pass over the whole split."""
    totals = {"raw": [0.0, 0.0], "std": [0.0, 0.0]}
    count = 0
    ds = load_split(cfg, "train", pretrain=True).map(lambda img, _: img).batch(1000)
    for batch in ds:
        raw = np.asarray(batch, np.float64) / 255.0
        out = np.asarray(standardise(batch, cfg), np.float64)
        for key, x in (("raw", raw), ("std", out)):
            totals[key][0] += x.sum(axis=(0, 1, 2))
            totals[key][1] += (x ** 2).sum(axis=(0, 1, 2))
        count += raw.shape[0] * raw.shape[1] * raw.shape[2]
    stats = {}
    for key, (s, sq) in totals.items():
        mean = s / count
        stats[key] = (mean, np.sqrt(sq / count - mean ** 2))
    return stats


# ---- the constants ---------------------------------------------------------

def test_normalisation_constants_match_the_dataset(cfg, pixel_stats):
    """config.mean and config.std are the actual statistics of the split."""
    mean, std = pixel_stats["raw"]

    assert np.allclose(mean, cfg.mean, atol=1e-4)
    assert np.allclose(std, cfg.std, atol=1e-4)


# ---- standardise -----------------------------------------------------------

def test_standardise_centres_and_scales_the_real_data(pixel_stats):
    mean, std = pixel_stats["std"]

    assert np.allclose(mean, 0.0, atol=1e-3)
    assert np.allclose(std, 1.0, atol=1e-3)


def test_standardise_is_the_exact_affine_map(cfg):
    """An all-255 image maps to (1.0 - mean) / std."""
    white = np.full((1, 4, 4, 3), 255, np.uint8)

    out = np.asarray(standardise(white, cfg))
    expected = (1.0 - np.array(cfg.mean)) / np.array(cfg.std)

    assert np.allclose(out[0, 0, 0], expected, atol=1e-5)
    assert np.allclose(out, expected, atol=1e-5), "a constant image stays constant"


def test_standardise_returns_float32(cfg):
    out = standardise(np.zeros((1, 4, 4, 3), np.uint8), cfg)

    assert out.dtype == "float32"


# ---- loading ---------------------------------------------------------------

def test_the_pipeline_sees_uint8_images_of_the_configured_size(cfg, first_image):
    assert tuple(first_image.shape) == (cfg.image_size, cfg.image_size, cfg.num_channels)
    assert first_image.dtype == tf.uint8


def test_stl10_images_are_96x96_before_the_resize():
    import tensorflow_datasets as tfds

    for split in ("unlabelled", "train", "test"):
        image, _ = next(iter(tfds.load("stl10", split=split, as_supervised=True)))
        assert tuple(image.shape) == (96, 96, 3), split


def test_stl10_pretrains_on_the_unlabelled_split():
    cfg = JEPAConfig(**DATASETS["stl10"])

    _, label = next(iter(load_split(cfg, "train", pretrain=True)))
    pretrain_size = int(load_split(cfg, "train", pretrain=True).cardinality())
    probe_size = int(load_split(cfg, "train", pretrain=False).cardinality())

    assert int(label) == -1, "the unlabelled split has label -1"
    assert (pretrain_size, probe_size) == (100_000, 5_000)


# ---- augment ---------------------------------------------------------------

def test_augment_preserves_shape_and_dtype(cfg, first_image):
    out = augment(first_image, cfg)

    assert tuple(out.shape) == (cfg.image_size, cfg.image_size, cfg.num_channels)
    assert out.dtype == first_image.dtype


def test_augment_only_moves_pixels_it_was_given(cfg, first_image):
    """Pad, crop and flip never create new pixel values."""
    original = set(np.asarray(first_image).ravel().tolist())

    for _ in range(20):
        produced = set(np.asarray(augment(first_image, cfg)).ravel().tolist())
        assert produced <= original


def test_augment_actually_augments(cfg, first_image):
    reference = np.asarray(first_image)

    changed = sum(
        not np.array_equal(np.asarray(augment(first_image, cfg)), reference)
        for _ in range(20)
    )
    assert changed > 0, "augment returned the input image 20 times in a row"


# ---- the datasets ----------------------------------------------------------

@pytest.fixture(scope="module")
def pretrain_ds(cfg):
    return build_pretrain_dataset(cfg, "train")


@pytest.fixture(scope="module")
def probe_test_ds(cfg):
    return build_probe_dataset(cfg, "test")


def test_pretrain_dataset_yields_unlabelled_batches(cfg, pretrain_ds):
    batch = next(iter(pretrain_ds))

    assert not isinstance(batch, tuple), "pretraining must not carry labels"
    assert tuple(batch.shape) == (
        cfg.batch_size, cfg.image_size, cfg.image_size, cfg.num_channels
    )
    assert batch.dtype == "float32"


def test_the_number_of_steps_per_epoch_is_known(cfg, pretrain_ds):
    """train.py reads it from the cardinality."""
    assert int(pretrain_ds.cardinality()) > 0


def test_pretrain_batches_differ_between_epochs(cfg, pretrain_ds):
    first = np.asarray(next(iter(pretrain_ds)))
    second = np.asarray(next(iter(pretrain_ds)))

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("split", ["train", "test"])
def test_probe_dataset_yields_image_label_pairs(cfg, split):
    images, labels = next(iter(build_probe_dataset(cfg, split)))

    assert tuple(images.shape)[1:] == (
        cfg.image_size, cfg.image_size, cfg.num_channels
    )
    assert images.dtype == "float32"
    assert tuple(labels.shape) == (tuple(images.shape)[0],)
    assert 0 <= int(np.min(labels)) and int(np.max(labels)) <= 9


def test_the_test_split_is_never_augmented(cfg, probe_test_ds):
    """The test split equals the raw test images through standardise only."""
    raw, _ = next(iter(load_split(cfg, "test", pretrain=False).batch(cfg.batch_size)))

    images, _ = next(iter(probe_test_ds))
    expected = np.asarray(standardise(raw, cfg))

    assert np.array_equal(np.asarray(images), expected), (
        "the test split went through something other than standardise, "
        "or its order changed"
    )


def test_the_test_split_is_stable_across_passes(cfg, probe_test_ds):
    first = np.asarray(next(iter(probe_test_ds))[0])
    second = np.asarray(next(iter(probe_test_ds))[0])

    assert np.array_equal(first, second)
