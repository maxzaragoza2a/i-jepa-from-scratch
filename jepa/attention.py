"""Multi-head self-attention.

Shapes (embed_dim=128, num_heads=4):
    (B, N, 128) -> q, k, v -> (B, 4, N, 32) -> weights (B, 4, N, N)
    -> (B, 4, N, 32) -> (B, N, 128)

Nothing depends on N, since the context length changes at every draw.
"""

import keras
from keras import ops


class MultiHeadSelfAttention(keras.layers.Layer):
    """(B, N, embed_dim) -> (B, N, embed_dim)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} is not divisible by num_heads "
                f"{num_heads}: the heads would cover "
                f"{num_heads * (embed_dim // num_heads)} dimensions instead "
                f"of {embed_dim}"
            )

        self.head_dim = self.embed_dim // num_heads

        self.query = keras.layers.Dense(embed_dim, name="query")
        self.key = keras.layers.Dense(embed_dim, name="key")
        self.value = keras.layers.Dense(embed_dim, name="value")
        self.projection = keras.layers.Dense(embed_dim, name="projection")
        self.attention_dropout = keras.layers.Dropout(dropout)

        # Attention weights of the last call (before dropout).
        self.last_attention_weights = None

    def split_heads(self, x):
        """(B, N, embed_dim) -> (B, num_heads, N, head_dim)."""
        B, N, _ = ops.shape(x)

        x = ops.reshape(x, (B, N, self.num_heads, self.head_dim))

        return ops.transpose(x, (0, 2, 1, 3))

    def merge_heads(self, x):
        """(B, num_heads, N, head_dim) -> (B, N, embed_dim)."""
        B, _, N, _ = ops.shape(x)

        x = ops.transpose(x, (0, 2, 1, 3))

        return ops.reshape(x, (B, N, self.embed_dim))

    def call(self, x, training=None):
        """(B, N, embed_dim) -> (B, N, embed_dim)."""
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))

        # (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(self.head_dim, scores.dtype))

        weights = ops.softmax(scores, axis=-1)
        self.last_attention_weights = weights

        out = ops.matmul(self.attention_dropout(weights, training=training), v)

        return self.projection(self.merge_heads(out))

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        })
        return config
