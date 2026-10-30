"""Tests for the package's public API surface and CLI entry points.

Requires torch and zstandard:

    uv run pytest tests/test_api.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


def _zstd():
    """zstandard is only needed by the dataset CLI test."""
    return pytest.importorskip("zstandard", reason="zstd CLI test needs zstandard")

SRC = str(Path(__file__).resolve().parents[1] / "src")


class TestLazyExports:
    """The top-level package must expose the public names without importing
    the whole ML stack eagerly -- ``myllm.dataset`` is usable with only
    ``zstandard`` installed."""

    def test_public_names_resolve(self):
        sys.path.insert(0, SRC)
        import myllm

        for name in [
            "Tokenizer",
            "ModelConfig",
            "LinearRNN",
            "LinearRNNConfig",
            "RowStochastic",
            "prefix_scan",
            "ZstLineDataset",
            "ZstCsvDataset",
            "ZstJsonLinesDataset",
            "train",
        ]:
            assert getattr(myllm, name) is not None, f"myllm.{name} missing"

    def test_unknown_attribute_raises(self):
        sys.path.insert(0, SRC)
        import myllm

        with pytest.raises(AttributeError):
            _ = myllm.definitely_not_a_real_name

    def test_dataset_importable_without_torch(self):
        """Importing myllm.dataset must not pull in torch/numpy."""
        code = (
            "import sys;"
            f"sys.path.insert(0, {SRC!r});"
            "import myllm.dataset;"
            "assert 'torch' not in sys.modules, 'torch was imported eagerly';"
            "assert 'numpy' not in sys.modules, 'numpy was imported eagerly';"
            "print('ok')"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
        assert "ok" in r.stdout


class TestCli:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "myllm.cli", *args],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
            check=False,
        )

    def test_help_lists_dataset_and_rnn_demo(self):
        r = self._run("--help")
        assert r.returncode == 0, r.stderr
        assert "dataset" in r.stdout
        assert "rnn-demo" in r.stdout

    def test_rnn_demo_runs(self):
        r = self._run("rnn-demo", "--hidden-size", "8", "--seq-len", "4")
        assert r.returncode == 0, r.stderr
        assert "spectral radius: 1.000000" in r.stdout
        assert "grad finite  : True" in r.stdout

    def test_rnn_demo_without_row_stochastic(self):
        r = self._run("rnn-demo", "--no-row-stochastic")
        assert r.returncode == 0, r.stderr
        assert "row_stochastic: False" in r.stdout

    def test_dataset_reports_missing_file(self):
        r = self._run("dataset", "/nonexistent/path.zst")
        assert r.returncode == 2
        assert "not found" in r.stdout.lower()

    def test_dataset_prints_records(self, tmp_path):
        zstd = _zstd()
        p = tmp_path / "d.zst"
        p.write_bytes(zstd.ZstdCompressor().compress(b"alpha\nbeta\ngamma\n"))
        r = self._run("dataset", str(p), "--num", "2")
        assert r.returncode == 0, r.stderr
        assert "records     : 3" in r.stdout
        assert "alpha" in r.stdout and "beta" in r.stdout
