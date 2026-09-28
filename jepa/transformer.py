"""Transformer block and ViT encoder.

The same encoder is used as context encoder (called with indices) and as
target encoder (called without). Tokens are gathered after the positional
encoding is added.
"""

import keras

from jepa.attention import MultiHeadSelfAttention
from jepa.masking import gather_tokens
from jepa.patch_embed import PatchEmbedding


class TransformerBlock(keras.layers.Layer):
    """Pre-norm transformer block: (B, N, embed_dim) -> (B, N, embed_dim).

        x = x + attention(norm1(x))
        x = x + mlp(norm2(x))
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout, name="attention")
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.mlp = keras.Sequential([
            keras.layers.Dense(int(embed_dim * mlp_ratio), activation="gelu", name="mlp_dense1"),
            keras.layers.Dropout(dropout, name="mlp_dropout1"),
            keras.layers.Dense(embed_dim, name="mlp_dense2"),
            keras.layers.Dropout(dropout, name="mlp_dropout2"),
        ], name="mlp")


    def call(self, x, training=None):
        """(B, N, embed_dim) -> (B, N, embed_dim)."""
        x = x + self.attention(self.norm1(x), training=training)
        return x + self.mlp(self.norm2(x), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
        })
        return config


class VisionTransformer(keras.layers.Layer):
    """(B, H, W, C) images -> (B, M, embed_dim) token embeddings.

    M is the number of indices passed in, or every patch when indices is None.
    """

    def __init__(self, patch_size: int, embed_dim: int, depth: int,
                 num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.patch_embedding = PatchEmbedding(patch_size, embed_dim, name="patch_embedding")
        self.blocks = [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, name=f"block_{i}") for i in range(depth)]
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")


    def call(self, images, indices=None, training=None):
        """Embed, optionally keep only indices, then run the blocks.

        Args:
            images: (B, H, W, C), standardised.
            indices: 1-D int array of token indices to keep, or None to keep
                all of them.
        """
        tokens = self.patch_embedding(images)
        if indices is not None:
            tokens = gather_tokens(tokens, indices)

        for block in self.blocks:
            tokens = block(tokens, training=training)
        tokens = self.norm(tokens)
        return tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
        })
        return config
