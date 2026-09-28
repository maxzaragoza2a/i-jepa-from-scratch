"""Predictor: context embeddings -> predicted target embeddings.

Shapes for one target block (M context tokens, T target tokens):
    (B, M, 128) -> Dense -> (B, M, 64)
    + T mask tokens (learned vector + positional code) -> (B, M + T, 64)
    -> blocks, norm -> last T tokens -> Dense -> (B, T, 128)
"""

import keras

from jepa.transformer import TransformerBlock


class Predictor(keras.layers.Layer):
    """(B, M, embed_dim) context tokens -> (B, T, embed_dim) predictions."""

    def __init__(self, grid_size: int, embed_dim: int, predictor_embed_dim: int,
                 depth: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, **kwargs):
        """
        Args:
            grid_size: patches per side.
            embed_dim: encoder width, input and output width.
            predictor_embed_dim: width inside the predictor.
            depth: number of TransformerBlocks.
            num_heads: attention heads, must divide predictor_embed_dim.
        """
        super().__init__(**kwargs)
        # TODO: store arguments, input Dense, blocks + norm, output Dense.
        raise NotImplementedError

    def build(self, input_shape):
        """Create mask_token (1, 1, predictor_embed_dim), learned, and
        pos_embed (num_patches, predictor_embed_dim), fixed."""
        # TODO: set pos_embed through the initializer, not assign().
        raise NotImplementedError

    def call(self, context_tokens, context_indices, target_indices, training=None):
        """Predict the target encoder's embeddings at target_indices.

        Args:
            context_tokens: (B, M, embed_dim), context encoder output.
            context_indices: 1-D int array, the M grid positions of the context.
            target_indices: 1-D int array, the T grid positions to predict.

        Returns:
            (B, T, embed_dim).
        """
        # TODO:
        #   1. context: Dense 128 -> 64 (+ position?)
        #   2. mask tokens: broadcast mask_token to (B, T, 64), add positions
        #   3. concatenate, run the blocks, norm
        #   4. keep the last T tokens, Dense 64 -> 128
        raise NotImplementedError

    def get_config(self):
        # TODO
        raise NotImplementedError
