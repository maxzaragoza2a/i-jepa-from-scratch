"""Hyperparameters for the project."""

from dataclasses import dataclass


@dataclass
class JEPAConfig:
    # ---- Data -------------------------------------------------------------
    image_size: int = 32          # CIFAR-10 is 32x32
    num_channels: int = 3
    patch_size: int = 4           # 8x8 grid, 64 patches per image

    # Per-channel CIFAR-10 statistics.
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)

    # ---- Encoder (shared by context and target) ---------------------------
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # ---- Predictor --------------------------------------------------------
    # Narrow and shallow, as in I-JEPA.
    predictor_embed_dim: int = 64
    predictor_depth: int = 2
    predictor_num_heads: int = 4

    # ---- Masking ----------------------------------------------------------
    num_target_blocks: int = 4
    target_scale: tuple = (0.15, 0.20)        # fraction of the grid per target
    target_aspect_ratio: tuple = (0.75, 1.5)  # height / width
    context_scale: tuple = (0.85, 1.00)

    # ---- Training ---------------------------------------------------------
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 3

    # EMA momentum of the target encoder, annealed to ema_end.
    ema_start: float = 0.996
    ema_end: float = 1.0

    seed: int = 42

    # ---- Derived shapes ---------------------------------------------------
    @property
    def grid_size(self) -> int:
        """Number of patches along one side."""
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} is not divisible by "
                f"patch_size {self.patch_size}"
            )
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total tokens per image."""
        return self.grid_size ** 2

    @property
    def patch_dim(self) -> int:
        """Flattened size of one raw patch."""
        return self.patch_size * self.patch_size * self.num_channels
