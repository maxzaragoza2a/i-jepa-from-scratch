"""Tests for multi-head self-attention."""

import numpy as np
import pytest

from jepa.attention import MultiHeadSelfAttention


@pytest.fixture
def mha():
    return MultiHeadSelfAttention(embed_dim=128, num_heads=4)


# ---- splitting and merging the heads ---------------------------------------

def test_split_heads_keeps_each_token_whole():
    """Each head gets a contiguous slice of every token."""
    layer = MultiHeadSelfAttention(embed_dim=4, num_heads=2)
    x = np.array([[[0.0, 1, 2, 3], [10, 11, 12, 13]]])

    heads = np.array(layer.split_heads(x))

    assert heads.shape == (1, 2, 2, 2)
    assert heads[0, 0].tolist() == [[0, 1], [10, 11]]
    assert heads[0, 1].tolist() == [[2, 3], [12, 13]]


@pytest.mark.parametrize("batch,tokens", [(1, 1), (2, 15), (3, 64)])
def test_merge_heads_undoes_split_heads(mha, batch, tokens):
    x = np.random.default_rng(0).standard_normal((batch, tokens, 128)).astype("float32")

    assert np.array_equal(np.array(mha.merge_heads(mha.split_heads(x))), x)


def test_embed_dim_must_divide_into_the_heads():
    with pytest.raises(ValueError, match="not divisible by num_heads"):
        MultiHeadSelfAttention(embed_dim=128, num_heads=5)


# ---- the attention weights themselves --------------------------------------

@pytest.mark.parametrize("tokens", [1, 15, 64])
def test_every_token_spends_its_whole_attention_budget(mha, tokens):
    """Attention rows sum to 1."""
    x = np.random.default_rng(1).standard_normal((2, tokens, 128)).astype("float32")

    mha(x)
    weights = np.array(mha.last_attention_weights)

    assert weights.shape == (2, 4, tokens, tokens)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=1e-5)


def test_the_stored_weights_are_the_ones_before_dropout():
    layer = MultiHeadSelfAttention(embed_dim=128, num_heads=4, dropout=0.5)
    x = np.random.default_rng(2).standard_normal((2, 16, 128)).astype("float32")

    layer(x, training=True)

    rows = np.array(layer.last_attention_weights).sum(axis=-1)
    np.testing.assert_allclose(rows, 1.0, rtol=1e-5)


# ---- permutation and mixing ------------------------------------------------

def test_permuting_the_tokens_permutes_the_output(mha):
    x = np.random.default_rng(3).standard_normal((1, 10, 128)).astype("float32")
    order = np.random.default_rng(3).permutation(10)

    shuffled_after = np.array(mha(x))[:, order]
    shuffled_before = np.array(mha(x[:, order]))

    np.testing.assert_allclose(shuffled_after, shuffled_before, atol=1e-5)


def test_tokens_really_do_read_each_other(mha):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 8, 128)).astype("float32")
    changed = x.copy()
    changed[0, 0] = rng.standard_normal(128)

    before = np.array(mha(x))
    after = np.array(mha(changed))

    assert not np.allclose(before[0, 3], after[0, 3], atol=1e-6)


def test_a_lone_token_can_only_look_at_itself(mha):
    x = np.random.default_rng(5).standard_normal((1, 1, 128)).astype("float32")

    mha(x)

    np.testing.assert_allclose(np.array(mha.last_attention_weights), 1.0)


# ---- independence from N ---------------------------------------------------

@pytest.mark.parametrize("tokens", [1, 7, 33, 64])
def test_one_layer_serves_any_sequence_length(mha, tokens):
    x = np.random.default_rng(6).standard_normal((2, tokens, 128)).astype("float32")

    assert np.array(mha(x)).shape == (2, tokens, 128)


def test_the_weights_do_not_depend_on_the_sequence_length(mha):
    mha(np.zeros((1, 5, 128), dtype="float32"))

    assert len(mha.weights) == 8
    assert {tuple(w.shape) for w in mha.weights} == {(128, 128), (128,)}


def test_dropout_is_off_outside_training():
    layer = MultiHeadSelfAttention(embed_dim=128, num_heads=4, dropout=0.5)
    x = np.random.default_rng(7).standard_normal((1, 16, 128)).astype("float32")

    assert np.array_equal(np.array(layer(x, training=False)),
                          np.array(layer(x, training=False)))
    assert not np.array_equal(np.array(layer(x, training=True)),
                              np.array(layer(x, training=True)))
