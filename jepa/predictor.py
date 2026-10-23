"""Predictor: context embeddings -> predicted target embeddings.

Shapes for one target block (M context tokens, T target tokens):
    (B, M, 256) -> Dense -> (B, M, 128)
    + T mask tokens (learned vector + positional code) -> (B, M + T, 128)
    -> blocks, norm -> last T tokens -> Dense -> (B, T, 256)
"""



import keras
from keras import ops
from jepa.transformer import TransformerBlock
from jepa.patch_embed import get_2d_sincos_pos_embed


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
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.input_proj = keras.layers.Dense(predictor_embed_dim, name="input_proj")
        self.blocks = [
            TransformerBlock(predictor_embed_dim, num_heads, mlp_ratio, dropout, name=f"block_{i}")
            for i in range(depth)
        ]
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.output_proj = keras.layers.Dense(embed_dim, name="output_proj")

    def build(self, input_shape):
        """Create mask_token (1, 1, predictor_embed_dim), learned, and
        pos_embed (num_patches, predictor_embed_dim), fixed."""
        num_patches = self.grid_size **2
        self.mask_token = self.add_weight(
            name="mask_token", shape=(1, 1, self.predictor_embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True
        )
        self.pos_embed = self.add_weight(
            name="pos_embed", shape=(num_patches, self.predictor_embed_dim),
            initializer=keras.initializers.Constant(get_2d_sincos_pos_embed(self.predictor_embed_dim, self.grid_size)), trainable=False
        )
        super().build(input_shape)

    def call(self, context_tokens, context_indices, target_indices, training=None):
        """Predict the target encoder's embeddings at target_indices.

        Args:
            context_tokens: (B, M, embed_dim), context encoder output.
            context_indices: 1-D int array, the M grid positions of the context.
            target_indices: 1-D int array, the T grid positions to predict.

        Returns:
            (B, T, embed_dim).
        """
        B = ops.shape(context_tokens)[0]
        M = ops.shape(context_tokens)[1]
        T = ops.shape(target_indices)[0]

        x = self.input_proj(context_tokens)
        pos_ctx = ops.take(self.pos_embed, context_indices, axis=0)
        ctx = x + pos_ctx

        x_target = ops.take(self.pos_embed, target_indices, axis=0)
        mask_tokens = self.mask_token + x_target
        mask_tokens = ops.broadcast_to(mask_tokens, (B, T, self.predictor_embed_dim))

        assembled = ops.concatenate([ctx, mask_tokens], axis=1)
        for block in self.blocks:
            assembled = block(assembled, training=training)
        assembled = self.norm(assembled)
        target_tokens = assembled[:, M:, :]
        return self.output_proj(target_tokens)

    def get_config(self):
        config = super().get_config()
        config.update({
            "grid_size": self.grid_size,
            "embed_dim": self.embed_dim,
            "predictor_embed_dim": self.predictor_embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
        })
        return config

