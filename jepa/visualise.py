"""Plot the context and target masks applied to training images.

Usage:

    python -m jepa.visualise
    python -m jepa.visualise --seed 42
    python -m jepa.visualise --run stl10_d256
    python -m jepa.visualise --out masks.png --no-show

Rows: input image, context patches only, target blocks over a dimmed image.
All images share the same mask, as in training.
"""

import argparse

import numpy as np

from jepa.config import DEFAULT_RUN, PRESETS, JEPAConfig, get_config
from jepa.data import build_pretrain_dataset
from jepa.masking import sample_masks

# Colour-blind friendly outline colours, one per target block.
BLOCK_COLOURS = ["#e66100", "#5d3a9b", "#1aff92", "#40b0a6", "#dc267f", "#ffb000"]

# Value used for hidden patches.
BLANK = 0.5

# Brightness factor for non-target areas in the targets row.
DIM = 0.25


def denormalise(images, config: JEPAConfig):
    """Undo standardise and clip to [0, 1]."""
    images = np.asarray(images)
    return np.clip(images * np.array(config.std) + np.array(config.mean), 0.0, 1.0)


def patch_region(index: int, config: JEPAConfig):
    """Flat token index -> (row slice, column slice) in pixels."""
    row, col = divmod(int(index), config.grid_size)
    p = config.patch_size
    return slice(row * p, (row + 1) * p), slice(col * p, (col + 1) * p)


def block_bounds(indices, config: JEPAConfig):
    """Bounding box of a target block in pixels: (left, top, width, height)."""
    rows, cols = np.divmod(np.asarray(indices), config.grid_size)
    p = config.patch_size
    top, left = rows.min() * p, cols.min() * p
    height = (rows.max() - rows.min() + 1) * p
    width = (cols.max() - cols.min() + 1) * p
    return left, top, width, height


def context_view(image, context_indices, config: JEPAConfig):
    """Image with everything outside the context blanked."""
    view = np.full_like(image, BLANK)
    for index in context_indices:
        rows, cols = patch_region(index, config)
        view[rows, cols] = image[rows, cols]
    return view


def target_view(image, target_indices, config: JEPAConfig):
    """Image dimmed everywhere except the target blocks."""
    view = image * DIM
    for block in target_indices:
        for index in block:
            rows, cols = patch_region(index, config)
            view[rows, cols] = image[rows, cols]
    return view


def plot_masked_samples(config=None, seed=None, num_images=5, out=None, show=True):
    """Draw one mask, apply it to num_images images and plot the result.

    Args:
        config: a JEPAConfig, default one if omitted.
        seed: seed of the draw; a random one is picked if None.
        num_images: number of images to show.
        out: path to save the figure to, or None.
        show: whether to open a window.

    Returns:
        The seed used.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    config = config or JEPAConfig()
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31))

    rng = np.random.default_rng(seed)

    batch = next(iter(build_pretrain_dataset(config, "train")))
    images = denormalise(batch, config)
    chosen = rng.choice(len(images), size=num_images, replace=False)

    context_indices, target_indices = sample_masks(config, rng)

    fig, axes = plt.subplots(3, num_images, figsize=(2.1 * num_images, 6.8))
    rows = ["input", f"context ({context_indices.size} patches)", "targets"]

    for column, image_index in enumerate(chosen):
        image = images[image_index]
        panels = [
            image,
            context_view(image, context_indices, config),
            target_view(image, target_indices, config),
        ]

        for row, panel in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(rows[row], fontsize=9)

        for block_number, block in enumerate(target_indices):
            left, top, width, height = block_bounds(block, config)
            axes[2, column].add_patch(Rectangle(
                (left - 0.5, top - 0.5), width, height,
                fill=False, linewidth=1.6,
                edgecolor=BLOCK_COLOURS[block_number % len(BLOCK_COLOURS)],
            ))

    hidden = np.unique(np.concatenate(target_indices)).size
    unused = config.num_patches - context_indices.size - hidden
    fig.suptitle(
        f"one mask, shared by the whole batch  --  seed {seed}\n"
        f"{context_indices.size} context, {hidden} hidden in "
        f"{len(target_indices)} blocks, {unused} unused  "
        f"(of {config.num_patches} patches)",
        fontsize=10,
    )
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"saved to {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN, choices=sorted(PRESETS),
                        help="preset of jepa/config.py, for the dataset and the grid")
    parser.add_argument("--seed", type=int, default=None,
                        help="reproduce a specific draw (default: random)")
    parser.add_argument("--images", type=int, default=5,
                        help="how many images to show (default: 5)")
    parser.add_argument("--out", default="masked_samples.png",
                        help="where to save the figure")
    parser.add_argument("--no-show", action="store_true",
                        help="save without opening a window")
    args = parser.parse_args()

    seed = plot_masked_samples(
        config=get_config(args.run),
        seed=args.seed,
        num_images=args.images,
        out=args.out,
        show=not args.no_show,
    )
    print(f"seed {seed} -- pass --seed {seed} to draw this mask again")


if __name__ == "__main__":
    main()
