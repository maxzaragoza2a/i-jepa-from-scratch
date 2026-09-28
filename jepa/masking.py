"""Context and target mask sampling on the patch grid.

Patches are numbered row-major: token i is at row i // grid_size, column
i % grid_size. Targets are rectangular blocks, and the context is one large
block with every target index removed from it.

One mask is drawn per batch and shared by all its images, so that the context
length is the same across the batch.
"""

import numpy as np
import tensorflow as tf


def sample_block(grid_size: int, scale: tuple, aspect_ratio: tuple, rng) -> tuple:
    """Draw one random rectangle inside a grid_size x grid_size grid.

    Args:
        grid_size: side of the patch grid.
        scale: (low, high) area as a fraction of the grid.
        aspect_ratio: (low, high) height / width.
        rng: numpy Generator.

    Returns:
        (top, left, height, width) in patch units.
    """
    area = rng.uniform(*scale) * grid_size**2
    ratio = rng.uniform(*aspect_ratio)
    height = int(round(np.sqrt(area * ratio)))
    width = int(round(np.sqrt(area / ratio)))
    height = np.clip(height, 1, grid_size)
    width = np.clip(width, 1, grid_size)
    top = rng.integers(0, grid_size - height + 1)
    left = rng.integers(0, grid_size - width + 1)
    return top, left, height, width


def block_to_indices(top: int, left: int, height: int, width: int,
                     grid_size: int) -> np.ndarray:
    """Flat row-major token indices covered by a rectangle, sorted ascending."""
    rows = np.arange(top, top + height)
    cols = np.arange(left, left + width)
    return (rows[:, None] * grid_size + cols[None, :]).reshape(-1)


def sample_masks(config, rng) -> tuple:
    """Draw one context / targets split, shared by a whole batch.

    Returns:
        context_indices: 1-D int array of the tokens given to the context
            encoder.
        target_indices: list of config.num_target_blocks 1-D int arrays, one
            per target block.
    """
    grid_size = config.grid_size

    target_indices = [
        block_to_indices(
            *sample_block(grid_size, config.target_scale,
                          config.target_aspect_ratio, rng),
            grid_size,
        )
        for _ in range(config.num_target_blocks)
    ]

    context_indices = block_to_indices(
        *sample_block(grid_size, config.context_scale, (1.0, 1.0), rng),
        grid_size,
    )

    # Remove every target index from the context.
    hidden = np.concatenate(target_indices)
    context_indices = np.setdiff1d(context_indices, hidden)

    return context_indices, target_indices


def gather_tokens(tokens, indices):
    """Select tokens at the given indices: (B, N, D) -> (B, M, D).

    The context is gathered before the context encoder; the targets are
    gathered after the target encoder has seen the full image.
    """
    return tf.gather(tokens, indices, axis=1)
