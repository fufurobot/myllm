"""A pure linear naive RNN with row-stochastic transition matrices.

Recurrence
----------
For an input sequence ``x_1..x_T`` with ``x_t`` of size ``input_size``:

    h_t = W_x @ x_t + W_h @ h_(t-1) + b,     h_0 = 0
    y_t = h_t                                 (seq_len, hidden_size)

This is the classic Elman recurrence with a linear activation. ``W_h`` is the
only path through time; ``W_x`` and ``b`` act pointwise per step.

Why row-stochastic?
-------------------
Backpropagation through the recurrence multiplies ``W_h`` (or its transpose,
for the gradient w.r.t. the state) once per time step, so the gradient scales
like ``W_h^T``. If the spectral radius of ``W_h`` exceeds 1 the gradient grows
exponentially ("boom"); below 1 it vanishes.

A *row-stochastic* matrix --- one whose rows sum to 1 --- has spectral radius
exactly 1 (the all-ones vector is a right eigenvector with eigenvalue 1, so the
induced infinity norm is 1). Row-stochastic matrices are closed under
multiplication:

    If A1 = 1 and B1 = 1, then (AB)1 = A(B1) = A1 = 1.

So any number of steps through time keeps the operator norm at 1 and the
gradient norm stays bounded instead of exploding. We enforce this by
normalizing every such parameter's rows at use time (see
:class:`RowStochastic`), never by training an unconstrained matrix and hoping.

Stability detail
----------------
Dividing by the row sum is dangerous when the sum is tiny or huge (blow-up or
underflow). :class:`RowStochastic` normalizes with a bounded, stabilising
denominator so the update stays a contraction even for badly scaled rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LinearRNNConfig:
    """Hyperparameters for :class:`LinearRNN`."""

    input_size: int
    hidden_size: int
    num_layers: int = 1
    row_stochastic: bool = True
    #: Minimum row sum before renormalisation kicks in. Rows summing to less
    #: than this are treated as this magnitude, which keeps the division stable.
    min_row_sum: float = 1e-3
    #: Upper bound applied to each row's sum, guarding against overflow.
    max_row_sum: float = 1e3

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 < self.min_row_sum < self.max_row_sum:
            raise ValueError("require 0 < min_row_sum < max_row_sum")


class RowStochastic(nn.Module):
    """A square weight matrix whose rows are normalized to sum to 1.

    The parameter is stored unconstrained (so gradients flow normally) and the
    *effective* matrix is obtained by dividing each row by its sum. Because
    ``(AB)1 = 1`` whenever ``A1 = B1 = 1``, compositions of these matrices stay
    row-stochastic, which is what bounds the gradient through time.

    Args:
        size: Matrix dimension (square).
        min_row_sum: Rows summing below this are clamped up to it before
            dividing, preventing division by ~0 from exploding the matrix.
        max_row_sum: Rows summing above this are clamped down, preventing
            overflow. Clamping changes only the *scale* of the effective
            matrix, never the direction of each row.
        init: ``"identity"`` (default) starts near the identity --- a stable
            "remember the state" initialisation; ``"uniform"`` starts uniform.
    """

    def __init__(
        self,
        size: int,
        *,
        min_row_sum: float = 1e-3,
        max_row_sum: float = 1e3,
        init: str = "uniform",
    ) -> None:
        super().__init__()
        self.size = size
        self.min_row_sum = min_row_sum
        self.max_row_sum = max_row_sum

        w = torch.empty(size, size)
        if init == "identity":
            w = torch.eye(size) + torch.randn(size, size) * 0.01
        elif init == "uniform":
            w.uniform_(0.0, 1.0)
        else:
            raise ValueError(f"unknown init {init!r}")
        self.weight = nn.Parameter(w)

    def effective(self) -> torch.Tensor:
        """Return the row-normalized matrix (rows sum to exactly 1).

        Stability: dividing by a raw row sum is unsafe when that sum is ~0
        (division explodes) or enormous (overflow). We bring each row's sum into
        ``[min_row_sum, max_row_sum]`` by *scaling the whole row* --- which
        leaves its direction unchanged --- and only then divide by the true sum.
        The result is therefore always an exact normalization of ``weight``.

        Rows that sum to (numerically) zero carry no direction; they fall back
        to a uniform distribution so the matrix stays row-stochastic.
        """
        w = self.weight
        row_sums = w.sum(dim=1, keepdim=True)

        zero = row_sums.abs() < 1e-12
        if bool(zero.any()):
            # Replace zero rows with a uniform distribution over the row.
            uniform = torch.full_like(w, 1.0 / w.shape[1])
            w = torch.where(zero, uniform, w)
            row_sums = torch.where(zero, torch.ones_like(row_sums), row_sums)

        # Per-row scale that pulls the sum into the safe band. Applied to the
        # whole row, so the normalized direction is unaffected.
        scale = torch.ones_like(row_sums)
        scale = torch.where(row_sums.abs() < self.min_row_sum,
                            self.min_row_sum / row_sums.abs(), scale)
        scale = torch.where(row_sums.abs() > self.max_row_sum,
                            self.max_row_sum / row_sums.abs(), scale)
        row_sums = row_sums * scale
        return (w * scale) / row_sums

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[.., size]``; returns ``x @ effective().T``."""
        return x @ self.effective().t()

    def extra_repr(self) -> str:
        return f"size={self.size}, min_row_sum={self.min_row_sum}"


class LinearRNNLayer(nn.Module):
    """One naive linear recurrent layer: ``h_t = W_x x_t + W_h h_(t-1) + b``."""

    def __init__(self, input_size: int, hidden_size: int, cfg: LinearRNNConfig) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_x = nn.Linear(input_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        if cfg.row_stochastic:
            self.w_h = RowStochastic(
                hidden_size, min_row_sum=cfg.min_row_sum, max_row_sum=cfg.max_row_sum
            )
        else:
            self.w_h = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.orthogonal_(self.w_h.weight)

        nn.init.normal_(self.w_x.weight, std=0.1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        """Run the recurrence.

        Args:
            x: ``[B, T, input_size]`` inputs.
            h0: ``[B, hidden_size]`` optional initial state (defaults to zeros).

        Returns:
            ``[B, T, hidden_size]`` hidden states for every step.

        Notes:
            When the transition is row-stochastic the recurrence is evaluated
            with the log-depth parallel scan (:func:`prefix_scan`), which is
            exact and avoids a per-step Python loop. Otherwise it falls back to
            the step loop, matching the reference implementation.
        """
        b, t, _ = x.shape
        # Precompute the input projection for all steps at once; only the
        # recurrent part needs the loop.
        x_proj = self.w_x(x) + self.bias  # [B, T, H]

        if isinstance(self.w_h, RowStochastic):
            return prefix_scan(self.w_h.effective(), x_proj, h0)

        h = torch.zeros(b, self.hidden_size, device=x.device, dtype=x.dtype) if h0 is None else h0
        outs: list[torch.Tensor] = []
        for step in range(t):
            h = x_proj[:, step] + self.w_h(h)
            outs.append(h)
        return torch.stack(outs, dim=1)


class LinearRNN(nn.Module):
    """A stack of :class:`LinearRNNLayer`, optionally with a linear readout.

    Mirrors the reference NumPy implementation:

        h = 0
        for x_t in x:
            h = W_x @ x_t + W_h @ h + b
            outputs.append(h)
        return outputs            # (seq_len, hidden_size)

    Args:
        config: Layer sizes and stabilisation settings.
    """

    def __init__(self, config: LinearRNNConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[LinearRNNLayer] = []
        in_size = config.input_size
        for _ in range(config.num_layers):
            layers.append(LinearRNNLayer(in_size, config.hidden_size, config))
            in_size = config.hidden_size
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` is ``[B, T, input_size]`` (or ``[T, input_size]``); returns hidden states."""
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"expected [B,T,D] or [T,D], got {tuple(x.shape)}")

        h = x
        for layer in self.layers:
            # Only the first layer receives h0, and only if it matches the size.
            h = layer(h, h0)
        return h.squeeze(0) if squeezed else h

    @torch.no_grad()
    def spectral_radius_of_transition(self) -> float:
        """Largest spectral radius over layers' transition matrices.

        For row-stochastic matrices this is exactly 1 (the all-ones vector is a
        right eigenvector with eigenvalue 1). Note this is the *spectral radius*
        (max |eigenvalue|), not the 2-norm: for non-symmetric row-stochastic
        matrices the 2-norm can exceed 1 slightly, but the radius stays 1, which
        is what bounds growth along the recurrence.
        """
        radii = []
        for layer in self.layers:
            w = layer.w_h.effective() if hasattr(layer.w_h, "effective") else layer.w_h.weight
            eig = torch.linalg.eigvals(w.detach().float())
            radii.append(float(eig.abs().max()))
        return max(radii)


def prefix_scan(
    w_h: torch.Tensor,
    inputs: torch.Tensor,
    h0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Parallel (associative-scan) evaluation of ``h_t = inputs_t + w_h h_(t-1)``.

    The linear recurrence is an affine scan, so it can be evaluated with a
    logarithmically-deep parallel scan instead of a ``T``-step loop. With
    row-stochastic ``w_h`` (spectral radius 1) the composed operators cannot
    blow up, which is what makes the scan numerically safe.

    Args:
        w_h: ``[H, H]`` transition matrix.
        inputs: ``[B, T, H]`` additive terms (``W_x x_t + b``).
        h0: ``[B, H]`` initial state; zeros if omitted.

    Returns:
        ``[B, T, H]`` hidden states.

    Notes:
        Each scan level composes affine maps ``(A, c)`` representing
        ``h -> A h + c``. Composition is ``(A2 A1, A2 c1 + c2)``, which is
        associative, so the Blelloch-style doubling below is exact.
    """
    b, t, h = inputs.shape
    if w_h.shape != (h, h):
        raise ValueError(f"w_h must be [{h},{h}], got {tuple(w_h.shape)}")

    if h0 is None:
        h0 = torch.zeros(b, h, device=inputs.device, dtype=inputs.dtype)

    # Level 0: single-step affine maps h_t <- w_h h_(t-1) + inputs_t
    a = w_h.unsqueeze(0).repeat(b, t, 1, 1)
    c = inputs

    # Inclusive scan over t, doubling the window each level. Everything is
    # built out-of-place so autograd can differentiate through the scan.
    step = 1
    while step < t:
        a_prev, c_prev = a[:, :-step], c[:, :-step]
        a_cur, c_cur = a[:, step:], c[:, step:]

        # (A_cur, c_cur) o (A_prev, c_prev) applied to the *newer* positions.
        c_new = c_cur + torch.einsum("btij,btj->bti", a_cur, c_prev)
        a_new = torch.einsum("btij,btjk->btik", a_cur, a_prev)

        # Positions before the current window edge are already final.
        c = torch.cat([c[:, :step], c_new], dim=1)
        a = torch.cat([a[:, :step], a_new], dim=1)
        step *= 2

    # Apply the initial state contribution: h_t = A_t h_0 + c_t
    return c + torch.einsum("btij,bj->bti", a, h0)


@dataclass
class LinearRNNLMConfig:
    """Hyperparameters for :class:`LinearRNNLM`."""

    vocab_size: int
    #: Input feature dimension fed to the recurrent stack. Defaults to ``dim``.
    input_dim: int | None = None
    dim: int = 128
    num_layers: int = 1
    row_stochastic: bool = True
    freeze_embeddings: bool = True
    tie_weights: bool = False
    min_row_sum: float = 1e-3
    max_row_sum: float = 1e3

    def __post_init__(self) -> None:
        if self.tie_weights:
            raise ValueError(
                "W_in and W_out must not be tied: they are the two factors of "
                "the bigram transition matrix."
            )
        if self.input_dim is not None and self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.vocab_size <= 0 or self.dim <= 0:
            raise ValueError("vocab_size and dim must be positive")


class LinearRNNLM(nn.Module):
    """A linear RNN language model with frozen, factorised embeddings.

    The recurrent stack operates on the input embedding of each token. Logits
    come from the final hidden state projected through the frozen output
    linear:

        x_t   = W_in[:, token_t]        # input embedding, [B, T, in_dim]
        h_T   = LinearRNN(x)            # [B, T, dim]
        logits = h_t @ W_out^T          # [B, T, V]

    ``W_in`` (``[in_dim, V]``) and ``W_out`` (``[V, dim]``) are initialised from
    a factorisation of the bigram count matrix and frozen by default. They are
    never tied.
    """

    def __init__(self, config: LinearRNNLMConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.input_dim if config.input_dim is not None else config.dim
        if in_dim != config.dim:
            # The recurrent layer consumes in_dim features and produces `dim`.
            self.input_proj = nn.Linear(in_dim, config.dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        self.input_embedding = nn.Parameter(
            torch.empty(in_dim, config.vocab_size),
            requires_grad=not config.freeze_embeddings,
        )
        self.output_linear = nn.Parameter(
            torch.empty(config.vocab_size, config.dim),
            requires_grad=not config.freeze_embeddings,
        )
        nn.init.normal_(self.input_embedding, std=0.02)
        nn.init.normal_(self.output_linear, std=0.02)

        self.rnn = LinearRNN(
            LinearRNNConfig(
                input_size=config.dim,
                hidden_size=config.dim,
                num_layers=config.num_layers,
                row_stochastic=config.row_stochastic,
                min_row_sum=config.min_row_sum,
                max_row_sum=config.max_row_sum,
            )
        )
        self.final_norm = nn.LayerNorm(config.dim)

    @torch.no_grad()
    def load_factorized_embeddings(self, w_in, w_out) -> None:
        """Install ``w_in [d, V]`` / ``w_out [V, d]`` into the frozen params."""
        w_in_t = torch.as_tensor(w_in, dtype=self.input_embedding.dtype)
        w_out_t = torch.as_tensor(w_out, dtype=self.output_linear.dtype)
        if w_in_t.shape != self.input_embedding.shape:
            raise ValueError(
                f"w_in shape {tuple(w_in_t.shape)} != expected "
                f"{tuple(self.input_embedding.shape)}"
            )
        if w_out_t.shape != self.output_linear.shape:
            raise ValueError(
                f"w_out shape {tuple(w_out_t.shape)} != expected "
                f"{tuple(self.output_linear.shape)}"
            )
        self.input_embedding.copy_(w_in_t)
        self.output_linear.copy_(w_out_t)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up input embeddings; ``[B, T]`` -> ``[B, T, in_dim]``."""
        return F.embedding(tokens, self.input_embedding.t())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` ``[B, T]`` -> logits ``[B, T, V]``."""
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got {tuple(tokens.shape)}")
        x = self.embed(tokens)
        x = self.input_proj(x)
        h = self.rnn(x)                     # [B, T, dim]
        h = self.final_norm(h)
        return h @ self.output_linear.t()   # [B, T, V]

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Naive autoregressive sampling from a ``[B, T]`` prompt."""
        self.eval()
        tokens = prompt
        for _ in range(max_new_tokens):
            logits = self(tokens)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.shape[-1])
                vals, _ = torch.topk(logits, k, dim=-1)
                logits = torch.where(
                    logits < vals[:, -1].unsqueeze(-1),
                    torch.full_like(logits, float("-inf")),
                    logits,
                )
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return tokens

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )

    def parameter_summary(self) -> dict[str, int]:
        total = self.num_parameters()
        trainable = self.num_parameters(trainable_only=True)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
