"""Tests for the row-stochastic LinearRNN.

Requires torch. Run with:

    uv run pytest tests/test_linear_rnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myllm.linear_rnn import (
    LinearRNN,
    LinearRNNConfig,
    LinearRNNLayer,
    RowStochastic,
    prefix_scan,
)


class TestMatchesReference:
    def test_sequence_shape(self):
        """``[T, input]`` in -> ``[T, hidden]`` out, like the NumPy reference."""
        m = LinearRNN(LinearRNNConfig(input_size=3, hidden_size=4, row_stochastic=False))
        out = m(torch.randn(5, 3))
        assert tuple(out.shape) == (5, 4)

    def test_batched_shape(self):
        m = LinearRNN(LinearRNNConfig(input_size=3, hidden_size=4))
        out = m(torch.randn(2, 5, 3))
        assert tuple(out.shape) == (2, 5, 4)

    def test_manual_recurrence_equivalence(self):
        """Layer output must equal h_t = W_x x_t + W_h h_(t-1) + b computed by hand."""
        torch.manual_seed(0)
        layer = LinearRNNLayer(4, 6, LinearRNNConfig(input_size=4, hidden_size=6))
        x = torch.randn(1, 5, 4)

        with torch.no_grad():
            out = layer(x)
            w_h = layer.w_h.effective()
            x_proj = layer.w_x(x) + layer.bias
            h = torch.zeros(1, 6)
            outs = []
            for t in range(5):
                h = x_proj[:, t] + h @ w_h.t()
                outs.append(h)
            expected = torch.stack(outs, dim=1)

        assert torch.allclose(out, expected, atol=1e-5)


class TestRowStochastic:
    def test_rows_sum_to_one(self):
        r = RowStochastic(8)
        torch.manual_seed(0)
        sums = r.effective().sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_closure_under_multiplication(self):
        """Row-stochastic matrices are closed under multiplication (the design claim)."""
        a = RowStochastic(6).effective()
        b = RowStochastic(6).effective()
        sums = (a @ b).sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_spectral_radius_is_one(self):
        r = RowStochastic(10)
        eig = torch.linalg.eigvals(r.effective().detach())
        assert abs(float(eig.abs().max()) - 1.0) < 1e-4

    def test_tiny_rows_stay_finite(self):
        r = RowStochastic(6)
        with torch.no_grad():
            r.weight.mul_(1e-12)
        eff = r.effective()
        assert torch.isfinite(eff).all()
        assert torch.allclose(eff.sum(1), torch.ones(6), atol=1e-5)

    def test_huge_rows_stay_finite(self):
        r = RowStochastic(6)
        with torch.no_grad():
            r.weight.mul_(1e20)
        eff = r.effective()
        assert torch.isfinite(eff).all()
        assert torch.allclose(eff.sum(1), torch.ones(6), atol=1e-5)

    def test_zero_row_becomes_uniform(self):
        r = RowStochastic(4)
        with torch.no_grad():
            r.weight[0] = 0.0
        eff = r.effective()
        assert torch.allclose(eff.sum(1), torch.ones(4), atol=1e-5)

    def test_negative_weights_normalize(self):
        r = RowStochastic(5)
        with torch.no_grad():
            r.weight.uniform_(-1, 1)
        eff = r.effective()
        assert torch.isfinite(eff).all()
        assert torch.allclose(eff.sum(1), torch.ones(5), atol=1e-5)


class TestPrefixScan:
    @pytest.mark.parametrize("t", [1, 2, 3, 5, 8, 17, 64])
    def test_matches_loop(self, t):
        torch.manual_seed(0)
        h = 5
        w = RowStochastic(h).effective().detach()
        inp = torch.randn(2, t, h)
        h0 = torch.randn(2, h)

        h_seq = h0.clone()
        outs = []
        for step in range(t):
            h_seq = inp[:, step] + h_seq @ w.t()
            outs.append(h_seq)
        expected = torch.stack(outs, dim=1)

        assert torch.allclose(prefix_scan(w, inp, h0), expected, atol=1e-4)

    def test_zero_initial_state(self):
        torch.manual_seed(0)
        h = 4
        w = RowStochastic(h).effective().detach()
        inp = torch.randn(1, 6, h)
        assert torch.allclose(
            prefix_scan(w, inp, None),
            prefix_scan(w, inp, torch.zeros(1, h)),
            atol=1e-6,
        )

    def test_long_sequence_finite(self):
        torch.manual_seed(0)
        h = 8
        w = RowStochastic(h).effective().detach()
        out = prefix_scan(w, torch.randn(1, 1024, h), None)
        assert torch.isfinite(out).all()

    def test_backward_through_scan(self):
        torch.manual_seed(0)
        layer = LinearRNNLayer(3, 5, LinearRNNConfig(input_size=3, hidden_size=5))
        out = layer(torch.randn(1, 6, 3))
        out.sum().backward()
        grad = layer.w_h.weight.grad
        assert grad is not None and torch.isfinite(grad).all()


class TestGradientStability:
    def test_row_stochastic_bounds_gradient_growth(self):
        """The point of the design: gradients grow, but far slower than an
        unconstrained recurrent matrix (which explodes)."""

        def final_grad_norm(steps: int, stochastic: bool) -> float:
            torch.manual_seed(0)
            layer = LinearRNNLayer(
                8, 16, LinearRNNConfig(input_size=8, hidden_size=16, row_stochastic=stochastic)
            )
            if not stochastic:
                with torch.no_grad():
                    layer.w_h.weight.mul_(1.5)
            out = layer(torch.randn(1, steps, 8))
            out.sum().backward()
            return float(layer.w_x.weight.grad.norm())

        short = final_grad_norm(10, True)
        long = final_grad_norm(200, True)
        # Row-stochastic growth is bounded; allow generous headroom.
        assert torch.isfinite(torch.tensor(long))
        assert long < short * 1e4

        # The unconstrained version overflows to inf/nan for the same depth.
        unstable = final_grad_norm(200, False)
        assert not torch.isfinite(torch.tensor(unstable)) or unstable > 1e12
