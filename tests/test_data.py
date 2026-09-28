"""Tests for the CIFAR-10 input pipeline.

Statistics are accumulated in float64: a float32 accumulator is not precise
enough over the full train split. These tests read the real dataset.
"""

import numpy as np
import pytest

from jepa.config import JEPAConfig
from jepa.data import (
    build_pretrain_dataset,
    build_probe_dataset,
    load_cifar10,
    standardise,
)


@pytest.fixture(scope="session")
def cfg():
    return JEPAConfig()


@pytest.fixture(scope="session")
def cifar():
    return load_cifar10()


# ---- the constants ---------------------------------------------------------

def test_normalisation_constants_match_the_dataset(cfg, cifar):
    """config.mean and config.std are the actual CIFAR-10 statistics."""
    (x_train, _), _ = cifar
    pixels = x_train.astype(np.float64) / 255.0

    assert np.allclose(pixels.mean(axis=(0, 1, 2)), cfg.mean, atol=1e-4)
    assert np.allclose(pixels.std(axis=(0, 1, 2)), cfg.std, atol=1e-4)


# ---- standardise -----------------------------------------------------------

def test_standardise_centres_and_scales_the_real_data(cfg, cifar):
    (x_train, _), _ = cifar

    out = np.asarray(standardise(x_train, cfg))

    assert np.allclose(out.mean(axis=(0, 1, 2), dtype=np.float64), 0.0, atol=1e-3)
    assert np.allclose(out.std(axis=(0, 1, 2), dtype=np.float64), 1.0, atol=1e-3)


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


# ---- augment ---------------------------------------------------------------

def test_augment_preserves_shape_and_dtype(cfg, cifar):
    import tensorflow as tf
    from jepa.data import augment

    (x_train, _), _ = cifar
    image = tf.constant(x_train[0])

    out = augment(image, cfg)

    assert tuple(out.shape) == (cfg.image_size, cfg.image_size, cfg.num_channels)
    assert out.dtype == image.dtype


def test_augment_only_moves_pixels_it_was_given(cfg, cifar):
    """Pad, crop and flip never create new pixel values."""
    import tensorflow as tf
    from jepa.data import augment

    (x_train, _), _ = cifar
    image = tf.constant(x_train[0])
    original = set(np.asarray(image).ravel().tolist())

    for _ in range(20):
        produced = set(np.asarray(augment(image, cfg)).ravel().tolist())
        assert produced <= original


def test_augment_actually_augments(cfg, cifar):
    import tensorflow as tf
    from jepa.data import augment

    (x_train, _), _ = cifar
    image = tf.constant(x_train[0])
    reference = np.asarray(image)

    changed = sum(
        not np.array_equal(np.asarray(augment(image, cfg)), reference)
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


def test_the_test_split_is_never_augmented(cfg, cifar, probe_test_ds):
    """The test split equals the raw test images through standardise only."""
    _, (x_test, _) = cifar

    images, _ = next(iter(probe_test_ds))
    expected = np.asarray(standardise(x_test[: cfg.batch_size], cfg))

    assert np.array_equal(np.asarray(images), expected), (
        "the test split went through something other than standardise, "
        "or its order changed"
    )


def test_the_test_split_is_stable_across_passes(cfg, probe_test_ds):
    first = np.asarray(next(iter(probe_test_ds))[0])
    second = np.asarray(next(iter(probe_test_ds))[0])

    assert np.array_equal(first, second)
