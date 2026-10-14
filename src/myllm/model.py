"""The TokenShift RNN language model.

Architecture
------------
A stack of identical recurrent layers. Each layer keeps a running state vector
and mixes it with a projection of that state using a gate.

Per layer, given the previous state ``s`` (``[B, d]``):

    out_emb    = W1 @ s^T                 -> [B, d]      (state readout)
    mix_factor = s . (W2 @ out_emb)       -> [B]         (scalar gate, dot product)
    gate       = sigmoid(mix_factor)      -> [B]
    next_state = gate * s + (1 - gate) * out_emb          -> [B, d]

The gate is a **dot product**, so it is a single scalar per batch element: the
layer interpolates between carrying the state forward and replacing it with the
state readout.

The initial state of layer 0 is the input embedding of the first token, which
matches "initial state = previous token" in the design.

Readout
-------
Logits come from the sum of every layer's ``out_emb``, projected through the
frozen output linear:

    logits = (sum_layers out_emb) @ W_out^T     -> [B, V]

``W_in`` (input embedding, ``[d, V]``) and ``W_out`` (output linear,
``[V, d]``) are initialised from a factorisation of the bigram count matrix and
are **frozen by default**; they are never tied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    """Hyperparameters for :class:`TokenShiftLM`."""

    vocab_size: int
    dim: int = 128
    num_layers: int = 2
    freeze_embeddings: bool = True
    tie_weights: bool = False  # deliberately False by design
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.tie_weights:
            raise ValueError(
                "W_in and W_out must not be tied: the design treats them as the "
                "two factors of the bigram transition matrix."
            )
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class TokenShiftLayer(nn.Module):
    """One recurrent layer implementing the state-mixing recurrence."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        # W1: state -> state readout ("output embedding" of the layer).
        self.w1 = nn.Linear(dim, dim, bias=False)
        # W2: state readout -> gate pre-activation.
        self.w2 = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Small init keeps the gate near 0.5 at the start, which makes the
        # layer behave like a slow average rather than saturating immediately.
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the state by one step.

        Args:
            state: ``[B, d]`` previous state.

        Returns:
            ``(next_state, out_emb)``, both ``[B, d]``.
        """
        out_emb = self.w1(state)                       # [B, d]
        gate_logit = (self.w2(out_emb) * state).sum(-1)  # [B]  dot product
        gate = torch.sigmoid(gate_logit).unsqueeze(-1)   # [B, 1]
        next_state = gate * state + (1.0 - gate) * out_emb
        return self.dropout(next_state), out_emb


class TokenShiftLM(nn.Module):
    """Stacked TokenShift RNN with a frozen factorised readout."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        v, d = config.vocab_size, config.dim

        # Frozen factorised matrices. Stored as [d, V] and [V, d].
        self.input_embedding = nn.Parameter(torch.empty(d, v), requires_grad=not config.freeze_embeddings)
        self.output_linear = nn.Parameter(torch.empty(v, d), requires_grad=not config.freeze_embeddings)

        self.layers = nn.ModuleList(
            TokenShiftLayer(d, dropout=config.dropout) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(d)

        # Sensible default init; overwritten by the factorisation before training.
        nn.init.normal_(self.input_embedding, std=0.02)
        nn.init.normal_(self.output_linear, std=0.02)

    # -- initialisation ------------------------------------------------------

    @torch.no_grad()
    def load_factorized_embeddings(self, w_in, w_out) -> None:
        """Install ``w_in [d, V]`` / ``w_out [V, d]`` into the frozen parameters."""
        w_in_t = torch.as_tensor(w_in, dtype=self.input_embedding.dtype)
        w_out_t = torch.as_tensor(w_out, dtype=self.output_linear.dtype)
        if w_in_t.shape != self.input_embedding.shape:
            raise ValueError(
                f"w_in shape {tuple(w_in_t.shape)} != expected {tuple(self.input_embedding.shape)}"
            )
        if w_out_t.shape != self.output_linear.shape:
            raise ValueError(
                f"w_out shape {tuple(w_out_t.shape)} != expected {tuple(self.output_linear.shape)}"
            )
        self.input_embedding.copy_(w_in_t)
        self.output_linear.copy_(w_out_t)

    # -- forward -------------------------------------------------------------

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up input embeddings. ``tokens`` is ``[B, T]`` or ``[B]``."""
        # input_embedding is [d, V]; gather columns for the token ids.
        return F.embedding(tokens, self.input_embedding.t())

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        """Run the recurrence over ``tokens``.

        Args:
            tokens: ``[B, T]`` token ids.
            state: Optional ``[B, d]`` initial state for each layer.
            return_state: Also return the final per-layer states.

        Returns:
            ``logits`` ``[B, T, V]``, or ``(logits, states)`` if
            ``return_state`` is True.
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got shape {tuple(tokens.shape)}")

        b, t = tokens.shape
        d = self.config.dim
        device = tokens.device

        # Layer 0 starts from the embedding of the first token:
        # "initial state = previous token".
        if state is None:
            states = [self.embed(tokens[:, 0]) for _ in range(self.config.num_layers)]
        else:
            if state.shape != (self.config.num_layers, b, d):
                raise ValueError(
                    f"state must be [{self.config.num_layers}, {b}, {d}], got {tuple(state.shape)}"
                )
            states = [state[i] for i in range(self.config.num_layers)]

        logits_per_step: list[torch.Tensor] = []
        for step in range(t):
            # Signal entering the stack at this step: the embedding of the
            # current token. Layer 0's *state* already carries the previous
            # token (or the previous step's state), per the recurrence.
            incoming = self.embed(tokens[:, step])          # [B, d]
            step_out_sum = torch.zeros(b, d, device=device, dtype=incoming.dtype)

            for i, layer in enumerate(self.layers):
                # Layer 0 mixes the carried state with the incoming token
                # embedding; deeper layers take the readout of the layer below.
                prev = states[i] if i == 0 else (states[i] + incoming)
                next_state, out_emb = layer(prev)
                states[i] = next_state
                incoming = out_emb
                step_out_sum = step_out_sum + out_emb

            step_out_sum = self.final_norm(step_out_sum)
            # logits = hidden @ W_out^T, W_out is [V, d].
            logits_per_step.append(step_out_sum @ self.output_linear.t())

        logits = torch.stack(logits_per_step, dim=1)  # [B, T, V]
        if return_state:
            return logits, torch.stack(states, dim=0)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy/temperature sampling from a ``[B, T]`` prompt."""
        self.eval()
        tokens = prompt
        state = None

        # Prime the state on the prompt.
        if prompt.shape[1] > 0:
            _, state = self.forward(prompt, state=None, return_state=True)

        for _ in range(max_new_tokens):
            last = tokens[:, -1:]
            logits, state = self.forward(last, state=state, return_state=True)
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                k = min(top_k, step_logits.shape[-1])
                values, _ = torch.topk(step_logits, k, dim=-1)
                threshold = values[:, -1].unsqueeze(-1)
                step_logits = torch.where(
                    step_logits < threshold,
                    torch.full_like(step_logits, float("-inf")),
                    step_logits,
                )

            probs = F.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if eos_id is not None and bool((next_token == eos_id).all()):
                break

        return tokens

    # -- introspection -------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )

    def parameter_summary(self) -> dict[str, int]:
        total = self.num_parameters()
        trainable = self.num_parameters(trainable_only=True)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
