"""Count-matrix factorisation used to initialise the frozen embeddings.

The design intent
-----------------
A pure ``softmax(W_out @ W_in[:, x])`` model is a bigram predictor: the logits
for the next token depend only on the current token. If we want the *default*
behaviour of the network to be exactly that bigram predictor, then ``W_in`` and
``W_out`` should be a factorisation of the bigram transition matrix.

Concretely, let ``C[i, j]`` be how often token ``j`` follows token ``i``. With

    W_in  : [d, V]     (input embedding, column i is the embedding of token i)
    W_out : [V, d]     (output linear, row j scores token j)

then ``W_out @ W_in`` is a rank-``d`` approximation of the transition matrix,
and the model starts life as the best rank-``d`` bigram model.

Two factorisations are provided:

``svd``
    Works on the ``ln`` scale: ``A = ln P(next | prev)``, approximated with a
    truncated SVD. Squared error on the log-probability scale, so rare and
    frequent transitions are weighted comparably.

``nmf``
    Works directly on the raw counts. Multiplicative-update NMF minimises
    Kullback-Leibler divergence on non-negative data, which is the natural
    objective for counts. Both factors stay non-negative.

Important: the two matrices are **never tied**. ``W_in`` is ``[d, V]`` and
``W_out`` is ``[V, d]``; they are separate parameters even when initialised
from the same factorisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CountMatrix:
    """Bigram counts plus the vocabulary bookkeeping needed to use them."""

    counts: np.ndarray  # [V, V] float64, NOT normalised
    vocab_size: int

    @property
    def total(self) -> float:
        return float(self.counts.sum())

    def log_prob(self, add_k: float = 1.0) -> np.ndarray:
        """Row-normalised ``ln P(next | prev)`` with add-``k`` smoothing.

        Rows that never occur in the corpus are left at ``-inf`` beyond the
        smoothing mass; they contribute nothing to the factorisation in
        practice because the SVD only sees finite values.
        """
        smoothed = self.counts + add_k
        row_sums = smoothed.sum(axis=1, keepdims=True)
        probs = smoothed / row_sums
        return np.log(probs)

    def save(self, path, ) -> None:  # pragma: no cover - convenience only
        np.savez_compressed(path, counts=self.counts)


def build_count_matrix(
    sequences: list[list[int]],
    vocab_size: int,
    include_eos: bool = True,
) -> CountMatrix:
    """Accumulate bigram counts over a list of id sequences.

    Each sequence is treated as independent: the transition from the end of one
    sequence into the beginning of the next is not counted.

    Args:
        sequences: Token id sequences.
        vocab_size: Size of the vocabulary (matrix is ``V x V``).
        include_eos: Whether to count transitions into the end-of-sequence
            token. Sequences are expected to already end with ``<eos>``.
    """
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for seq in sequences:
        if len(seq) < 2:
            continue
        prev = np.asarray(seq[:-1], dtype=np.int64)
        nxt = np.asarray(seq[1:], dtype=np.int64)
        np.add.at(counts, (prev, nxt), 1.0)

    if not include_eos:
        pass  # counts already reflect whatever ids were passed in

    return CountMatrix(counts=counts, vocab_size=vocab_size)


def factorize_svd(count_matrix: CountMatrix, dim: int, add_k: float = 1.0):
    """Truncated SVD of ``ln P(next | prev)``.

    Returns ``(w_in, w_out)`` with shapes ``[dim, V]`` and ``[V, dim]`` such
    that ``w_out @ w_in`` approximates the log-probability matrix.
    """
    logp = count_matrix.log_prob(add_k=add_k)
    # Replace -inf (never-observed, zero-probability) entries with the smallest
    # finite value so the SVD stays well defined.
    finite = np.isfinite(logp)
    if not finite.all():
        fill = logp[finite].min() if finite.any() else 0.0
        logp = np.where(finite, logp, fill)

    # Center per row so the rank-d approximation focuses on the transition
    # structure instead of reproducing the marginal.
    row_mean = logp.mean(axis=1, keepdims=True)
    centered = logp - row_mean

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(dim, s.shape[0])

    u = u[:, :rank]
    s = s[:rank]
    vt = vt[:rank, :]

    sqrt_s = np.sqrt(s)
    # w_in: [d, V], w_out: [V, d]. Absorb the row mean into w_out by giving
    # w_in a constant feature, so the reconstruction keeps the marginal.
    w_in = sqrt_s[:, None] * vt          # [rank, V]
    w_out = u * sqrt_s[None, :]          # [V, rank]

    if rank < dim:
        w_in, w_out = _pad_to_dim(w_in, w_out, dim)

    return w_in.astype(np.float32), w_out.astype(np.float32)


def factorize_nmf(
    count_matrix: CountMatrix,
    dim: int,
    n_iter: int = 300,
    seed: int = 0,
    eps: float = 1e-9,
):
    """NMF of the raw count matrix using KL-divergence multiplicative updates.

    Solves ``C ~= W_out @ W_in`` with ``W_out [V, d]``, ``W_in [d, V]``, both
    non-negative. This is Lee & Seung's ``C ~= H @ W`` form with the KL
    objective; the update rules below are the standard ones for that objective.

    Working on raw counts (rather than logs) keeps the factorisation
    interpretable: each entry is a share of the observed count mass.
    """
    c = count_matrix.counts.astype(np.float64)
    v = c.shape[0]
    rng = np.random.default_rng(seed)

    # Scale-aware random init keeps the first updates well conditioned.
    scale = max(c.sum() / max(v * dim, 1), eps)
    h = rng.random((v, dim)) * scale + eps   # W_out
    w = rng.random((dim, v)) * scale + eps   # W_in

    ones = np.ones_like(c)

    for _ in range(n_iter):
        # --- update W (=W_in) with H (=W_out) fixed -------------------------
        wh = h @ w
        htw = h.T @ (c / (wh + eps))
        htt1 = h.T @ ones
        w *= htw / (htt1 + eps)

        # --- update H (=W_out) with W (=W_in) fixed -------------------------
        wh = h @ w
        cwt = (c / (wh + eps)) @ w.T
        onew = ones @ w.T
        h *= cwt / (onew + eps)

    if dim > h.shape[1]:
        h, w = _pad_to_dim(w, h, dim)

    return w.astype(np.float32), h.astype(np.float32)


def _pad_to_dim(w_in: np.ndarray, w_out: np.ndarray, dim: int):
    """Zero-pad the rank axis when the data yields fewer components than ``dim``."""
    rank = w_in.shape[0]
    if rank >= dim:
        return w_in, w_out
    pad = dim - rank
    w_in = np.concatenate([w_in, np.zeros((pad, w_in.shape[1]), dtype=w_in.dtype)], axis=0)
    w_out = np.concatenate([w_out, np.zeros((w_out.shape[0], pad), dtype=w_out.dtype)], axis=1)
    return w_in, w_out


def normalize_factors(w_in: np.ndarray, w_out: np.ndarray, eps: float = 1e-8):
    """Rescale factors so each row of ``w_out`` has unit norm.

    The product ``w_out @ w_in`` is preserved exactly: the scale is taken per
    *rank* component (the shared axis), so dividing it out of ``w_out`` is
    compensated by multiplying it into ``w_in``.

    ``w_in`` is ``[d, V]`` and ``w_out`` is ``[V, d]``; the rank axis is axis 0
    of ``w_in`` and axis 1 of ``w_out``.
    """
    # Per-rank-component magnitude, computed from w_out's columns [d].
    scale = np.linalg.norm(w_out, axis=0, keepdims=True)  # [1, d]
    scale = np.maximum(scale, eps)
    w_in_scaled = w_in * scale.T          # [d, V] * [d, 1] -> scales rank axis
    w_out_scaled = w_out / scale          # [V, d] / [1, d] -> scales rank axis
    return w_in_scaled.astype(np.float32), w_out_scaled.astype(np.float32)
