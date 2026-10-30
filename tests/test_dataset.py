"""Contract tests for the lazy ``.zst`` dataset loaders.

These tests need only ``zstandard`` (no torch), so they run in any environment
that has it installed:

    python3 -m pytest tests/test_dataset.py      # system python
    uv run pytest tests/test_dataset.py          # project venv

Covered contracts
-----------------
1. A ``.zst`` file is read record-per-line, lazily, without loading it whole.
2. ``len()``, positive/negative indexing, and slicing behave like a sequence.
3. ``ZstCsvDataset`` hides the header row from the data indices.
4. ``ZstJsonLinesDataset`` yields objects, and can extract a key.
5. The LRU is bounded and reports hits/misses.
6. **Indexed access must agree with the streaming order.**
7. **Sequential indexed access must not be quadratic** -- reading N records in
   order has to cost roughly one pass over the data, not N passes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

zstd = pytest.importorskip("zstandard", reason="zstd datasets require zstandard")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myllm.dataset import (
    LRUCache,
    ZstCsvDataset,
    ZstdUnavailableError,
    ZstJsonLinesDataset,
    ZstLineDataset,
)


@pytest.fixture()
def fixtures(tmp_path):
    """Write small multi-chunk .zst fixtures and return their descriptions."""
    cctx = zstd.ZstdCompressor(level=3)
    out = {}

    # Wide records so that 512-byte chunks force many chunks.
    lines = [f"record {i} value={i * 2} " + "pad" * 20 for i in range(2000)]
    p = tmp_path / "lines.zst"
    p.write_bytes(cctx.compress("\n".join(lines).encode()))
    out["lines"] = (p, lines)

    csv_body = "text,label\n" + "\n".join(f'"row {i}",{i}' for i in range(800))
    p = tmp_path / "data.csv.zst"
    p.write_bytes(cctx.compress(csv_body.encode()))
    out["csv"] = (p, 800)

    jsonl = "\n".join(json.dumps({"i": i, "text": f"t{i}"}) for i in range(300))
    p = tmp_path / "data.jsonl.zst"
    p.write_bytes(cctx.compress(jsonl.encode()))
    out["jsonl"] = (p, 300)

    return out


class TestSequenceProtocol:
    def test_len_and_first_last(self, fixtures):
        path, lines = fixtures["lines"]
        ds = ZstLineDataset(path)
        assert len(ds) == len(lines)
        assert ds[0] == lines[0]
        assert ds[-1] == lines[-1]

    def test_negative_indexing(self, fixtures):
        path, lines = fixtures["lines"]
        ds = ZstLineDataset(path)
        assert ds[-5] == lines[-5]
        assert ds[-len(lines)] == lines[0]

    def test_slicing(self, fixtures):
        path, lines = fixtures["lines"]
        ds = ZstLineDataset(path)
        assert ds[10:14] == lines[10:14]

    def test_out_of_range_raises(self, fixtures):
        path, _ = fixtures["lines"]
        ds = ZstLineDataset(path)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]
        with pytest.raises(IndexError):
            _ = ds[-len(ds) - 1]

    def test_skip_empty_records(self, tmp_path):
        p = tmp_path / "blank.zst"
        p.write_bytes(zstd.ZstdCompressor().compress(b"a\n\nb\n\n\nc\n"))
        ds = ZstLineDataset(p)
        assert ds[:] == ["a", "b", "c"]

    def test_directory_of_files_is_concatenated(self, tmp_path):
        c = zstd.ZstdCompressor()
        (tmp_path / "a.zst").write_bytes(c.compress(b"x1\nx2\n"))
        (tmp_path / "b.zst").write_bytes(c.compress(b"y1\ny2\n"))
        ds = ZstLineDataset(tmp_path)
        assert len(ds) == 4
        assert ds[0] == "x1" and ds[2] == "y1" and ds[3] == "y2"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ZstLineDataset(tmp_path / "nope.zst")


class TestIndexAgreesWithStream:
    """Contract 6: random access must return exactly what streaming returns."""

    def test_indexed_matches_stream(self, fixtures):
        path, _ = fixtures["lines"]
        ds = ZstLineDataset(path, chunk_size=512, cache_size=2)
        streamed = list(ds.iter_raw_lines())
        for i in range(0, len(streamed), 37):
            assert ds[i] == streamed[i], f"mismatch at {i}"

    def test_backward_jump_after_forward_read(self, fixtures):
        path, lines = fixtures["lines"]
        ds = ZstLineDataset(path, chunk_size=512, cache_size=2)
        _ = ds[1500]  # force the cursor forward
        assert ds[3] == lines[3]  # must restart and still be correct
        assert ds[1500] == lines[1500]


class TestCsvDataset:
    def test_header_hidden_from_indices(self, fixtures):
        path, n = fixtures["csv"]
        ds = ZstCsvDataset(path)
        assert ds.fields == ["text", "label"]
        assert len(ds) == n
        assert ds[0] == {"text": "row 0", "label": "0"}

    def test_negative_index_skips_header(self, fixtures):
        path, n = fixtures["csv"]
        ds = ZstCsvDataset(path)
        assert ds[-1] == {"text": f"row {n - 1}", "label": str(n - 1)}

    def test_explicit_fields_mean_no_header_line(self, fixtures):
        path, _ = fixtures["csv"]
        ds = ZstCsvDataset(path, fields=["a", "b"], as_dict=False)
        assert ds[0] == ("text", "label")

    def test_positional_tuples(self, fixtures):
        path, _ = fixtures["csv"]
        ds = ZstCsvDataset(path, as_dict=False)
        assert ds[0] == ("row 0", "0")

    def test_past_end_raises(self, fixtures):
        path, n = fixtures["csv"]
        ds = ZstCsvDataset(path)
        with pytest.raises(IndexError):
            _ = ds[n]


class TestJsonLinesDataset:
    def test_objects(self, fixtures):
        path, n = fixtures["jsonl"]
        ds = ZstJsonLinesDataset(path)
        assert len(ds) == n
        assert ds[42] == {"i": 42, "text": "t42"}

    def test_key_extraction(self, fixtures):
        path, _ = fixtures["jsonl"]
        ds = ZstJsonLinesDataset(path, key="text")
        assert ds[7] == "t7"

    def test_negative_index(self, fixtures):
        path, n = fixtures["jsonl"]
        ds = ZstJsonLinesDataset(path)
        assert ds[-1]["i"] == n - 1


class TestLRU:
    def test_bounded_and_evicts_least_recent(self):
        c = LRUCache(size=2)
        c.put("a", 1)
        c.put("b", 2)
        assert c.get("a") == 1  # 'a' becomes most-recent
        c.put("c", 3)  # evicts 'b'
        assert c.get("b") is None
        assert c.get("a") == 1 and c.get("c") == 3
        assert len(c) <= 2

    def test_hit_rate_accounting(self):
        c = LRUCache(size=2)
        c.put("a", 1)
        c.get("a")
        c.get("missing")
        assert (c.hits, c.misses) == (1, 1)
        assert c.hit_rate == 0.5

    def test_rejects_nonpositive_size(self):
        with pytest.raises(ValueError):
            LRUCache(size=0)


class TestSequentialPerformance:
    """Contract 7: sequential indexed access must be ~linear, not quadratic.

    The dominant cost is *per-access work inside a chunk*. If ``__getitem__``
    re-filters the whole chunk on every access, reading N records that share a
    chunk costs O(N * chunk_size) instead of O(N). On a 20k-record fixture this
    is the difference between ~0.02s and ~2s for 5000 reads.

    The timing budget below is wide (~50x headroom over linear) but the
    re-filtering implementation cannot meet it.
    """

    @staticmethod
    def _write(tmp_path, n=20_000):
        cctx = zstd.ZstdCompressor(level=1)
        body = "\n".join(f"record {i} " + "x" * 40 for i in range(n))
        p = tmp_path / "perf.zst"
        p.write_bytes(cctx.compress(body.encode()))
        return p

    def test_chunks_decoded_once_per_scan(self, tmp_path):
        """A sequential scan must decode each chunk ~once, not once per record."""
        path = self._write(tmp_path)
        ds = ZstLineDataset(path, chunk_size=2048, cache_size=4)
        n = len(ds)
        n_chunks = sum(len(r) for r in ds._chunk_records)

        produced = {"count": 0}
        real_iter = ds._iter_decompressed_chunks

        def counting_iter(file_idx):
            for chunk in real_iter(file_idx):
                produced["count"] += 1
                yield chunk

        ds._iter_decompressed_chunks = counting_iter  # type: ignore[method-assign]

        for i in range(5_000):
            _ = ds[i]

        assert produced["count"] <= n_chunks * 3, (
            f"decoded {produced['count']} chunks while scanning 5000 of {n} "
            f"records over {n_chunks} chunks -- looks quadratic"
        )

    def test_sequential_reads_complete_quickly(self, tmp_path):
        path = self._write(tmp_path)
        ds = ZstLineDataset(path, chunk_size=2048, cache_size=4)
        _ = len(ds)

        start = time.perf_counter()
        for i in range(5_000):
            _ = ds[i]
        elapsed = time.perf_counter() - start

        # Observed: ~0.02s when a chunk is indexed directly, ~2.1s when the
        # chunk is re-filtered on every access. 1.0s separates them widely.
        assert elapsed < 1.0, (
            f"5000 sequential reads took {elapsed:.2f}s -- per-access work "
            "looks proportional to chunk size (e.g. re-filtering the chunk "
            "on every __getitem__)"
        )

    def test_cache_hit_rate_on_sequential_scan(self, tmp_path):
        """A sequential scan with a small cache still hits while inside a chunk."""
        cctx = zstd.ZstdCompressor(level=1)
        body = "\n".join(f"rec {i} " + "y" * 40 for i in range(5000))
        p = tmp_path / "scan.zst"
        p.write_bytes(cctx.compress(body.encode()))
        ds = ZstLineDataset(p, chunk_size=2048, cache_size=4)
        for i in range(500):
            _ = ds[i]
        assert ds.cache.hits > 0, "expected some cache hits during a sequential scan"


class TestMissingDependency:
    def test_clear_error_without_zstandard(self, tmp_path, monkeypatch):
        import myllm.dataset as m

        # The file must exist: path validation deliberately runs before the
        # dependency check so a missing file reports "not found".
        p = tmp_path / "whatever.zst"
        p.write_bytes(b"")
        monkeypatch.setattr(m, "zstd", None)
        with pytest.raises(ZstdUnavailableError):
            m.ZstLineDataset(p)

    def test_missing_file_reported_before_missing_dependency(self, tmp_path, monkeypatch):
        import myllm.dataset as m

        monkeypatch.setattr(m, "zstd", None)
        with pytest.raises(FileNotFoundError):
            m.ZstLineDataset(tmp_path / "absent.zst")
