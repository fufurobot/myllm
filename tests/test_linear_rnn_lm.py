"""Tests for LinearRNNLM and the zst-backed training path.

The trainer must build its model from ``LinearRNNLM`` (linear recurrence +
prefix scan + row-stochastic transitions), not the old TokenShift LM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zstandard")

SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)

from myllm.linear_rnn import LinearRNNLM, LinearRNNLMConfig  # noqa: E402


class TestLinearRNNLM:
    def test_logits_shape(self):
        m = LinearRNNLM(LinearRNNLMConfig(vocab_size=50, dim=16, num_layers=2))
        out = m(torch.randint(0, 50, (3, 9)))
        assert tuple(out.shape) == (3, 9, 50)

    def test_embeddings_frozen_by_default(self):
        m = LinearRNNLM(LinearRNNLMConfig(vocab_size=50, dim=16))
        assert m.input_embedding.requires_grad is False
        assert m.output_linear.requires_grad is False
        assert m.parameter_summary()["frozen"] > 0

    def test_unfreeze_embeddings(self):
        m = LinearRNNLM(
            LinearRNNLMConfig(vocab_size=50, dim=16, freeze_embeddings=False)
        )
        assert m.input_embedding.requires_grad is True

    def test_weights_not_tied(self):
        m = LinearRNNLM(LinearRNNLMConfig(vocab_size=50, dim=16))
        assert m.input_embedding.data_ptr() != m.output_linear.data_ptr()

    def test_tie_weights_rejected(self):
        with pytest.raises(ValueError):
            LinearRNNLMConfig(vocab_size=50, dim=16, tie_weights=True)

    def test_load_factorized_embeddings(self):
        import numpy as np

        m = LinearRNNLM(LinearRNNLMConfig(vocab_size=8, dim=4))
        w_in = np.random.randn(4, 8).astype("float32")
        w_out = np.random.randn(8, 4).astype("float32")
        m.load_factorized_embeddings(w_in, w_out)
        assert torch.allclose(m.input_embedding, torch.tensor(w_in))
        assert torch.allclose(m.output_linear, torch.tensor(w_out))

    def test_bad_shape_rejected(self):
        import numpy as np

        m = LinearRNNLM(LinearRNNLMConfig(vocab_size=8, dim=4))
        with pytest.raises(ValueError):
            m.load_factorized_embeddings(np.zeros((3, 8), "float32"), np.zeros((8, 4), "float32"))

    def test_transition_is_row_stochastic(self):
        m = LinearRNNLM(LinearRNNLMConfig(vocab_size=20, dim=8, num_layers=2))
        assert abs(m.rnn.spectral_radius_of_transition() - 1.0) < 1e-4

    def test_gradients_flow(self):
        m = LinearRNNLM(LinearRNNLMConfig(vocab_size=20, dim=8))
        m(torch.randint(0, 20, (2, 6))).sum().backward()
        g = m.rnn.layers[0].w_x.weight.grad
        assert g is not None and torch.isfinite(g).all()


class TestTrainerUsesLinearRNN:
    def test_trainer_imports_linear_rnn_lm(self):
        """The trainer must construct LinearRNNLM, not the TokenShift LM."""
        import myllm.trainer as t

        src = Path(t.__file__).read_text()
        assert "LinearRNNLM" in src, "trainer does not use LinearRNNLM"
        assert "TokenShiftLM(model_cfg)" not in src, (
            "trainer still constructs TokenShiftLM"
        )


class TestZstCorpusLoading:
    def test_load_lines_from_zst_csv(self, tmp_path):
        import zstandard as zstd

        from myllm.data import load_zst_corpus

        body = "sentence\n" + "\n".join(
            f'"verse number {i} has words"' for i in range(50)
        )
        p = tmp_path / "c.csv.zst"
        p.write_bytes(zstd.ZstdCompressor().compress(body.encode()))

        corpus = load_zst_corpus(p, text_field="sentence")
        assert len(corpus) == 50
        assert corpus.lines[0] == "verse number 0 has words"
        assert corpus.mode in {"word", "char"}

    def test_text_field_missing_raises(self, tmp_path):
        import zstandard as zstd

        from myllm.data import load_zst_corpus

        p = tmp_path / "c.csv.zst"
        p.write_bytes(zstd.ZstdCompressor().compress(b"a,b\n1,2\n"))
        with pytest.raises(KeyError):
            load_zst_corpus(p, text_field="sentence")
