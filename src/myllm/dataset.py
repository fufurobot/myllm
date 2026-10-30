"""Unified lazy dataset loaders over line-separated ``.zst`` files.

All loaders read Zstandard-compressed text where **one record per line** is the
storage unit. Depending on the subclass a line is either a raw value, a CSV row,
or a JSON object.

Design notes
------------
* **Lazy.** Files are never fully decompressed on open. A growing index maps
  record number -> ``(byte_offset, compressed_chunk_id)`` and only the needed
  chunk is decompressed. This keeps RAM flat even for multi-gigabyte corpora.
* **LRU chunk cache.** Decompressed chunks are cached in a bounded LRU, so
  sequential access decompresses each chunk once and random access stays cheap.
* **Thread-safe enough for DataLoaders.** ``num_workers>0`` in PyTorch forks
  the process; each worker therefore re-opens its own handle to the file and
  the LRU is per-process, which avoids sharing file offsets across processes.

Only ``zstandard`` is required; splitting/parsing of CSV and JSONL uses the
standard library.

Public API
----------
``ZstLineDataset``        base class, one record per line
``ZstCsvDataset``         CSV rows (``csv.DictReader`` or positional tuples)
``ZstJsonLinesDataset``   one JSON object per line
"""

from __future__ import annotations

import bisect
import csv
import io
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

try:  # pragma: no cover - exercised indirectly when the dep is present
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None  # type: ignore[assignment]


DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB of *compressed* input per read
DEFAULT_CACHE_SIZE = 32  # number of decompressed chunks held per dataset


class ZstdUnavailableError(RuntimeError):
    """Raised when a ``.zst`` dataset is opened without ``zstandard`` installed."""


def _require_zstd() -> None:
    if zstd is None:
        raise ZstdUnavailableError(
            "reading .zst datasets requires the 'zstandard' package.\n"
            "Install it with:  uv pip install zstandard"
        )


class LRUCache:
    """A small bounded LRU cache, thread-safe.

    Used for decompressed chunks. ``size`` counts entries, not bytes; chunk
    size is bounded by ``chunk_size`` so the memory ceiling is roughly
    ``size * chunk_size`` (decompressed).
    """

    def __init__(self, size: int = DEFAULT_CACHE_SIZE) -> None:
        if size <= 0:
            raise ValueError("cache size must be positive")
        self.size = size
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class _Chunk:
    """A line-aligned decompressed chunk plus the record range it covers."""

    text: list[str]
    start_record: int
    num_records: int


@dataclass
class _Cursor:
    """A forward-only streaming position over one file's chunks."""

    gen: Iterator[list[str]]
    next_id: int


class ZstLineDataset:
    """Lazy, indexed, line-per-record reader for ``.zst`` files.

    The dataset behaves like a ``Sequence[str]``: ``len()`` is the number of
    records and ``ds[i]`` returns the raw line (without the trailing newline).

    Args:
        path: ``.zst`` file containing newline-separated records.
            ``.zst`` files are also accepted if a directory is passed, in which
            case every ``*.zst`` inside is concatenated in sorted order.
        chunk_size: Compressed bytes read per I/O operation.
        cache_size: Number of decompressed chunks kept in the LRU.
        encoding: Text encoding of the decompressed payload.
        strip: Strip surrounding whitespace from each returned line.
        skip_empty: Drop blank lines from the index.
        index_all: If True, build the full index on open. Otherwise the index
            grows on demand as records are accessed sequentially, which makes
            opening a huge file effectively O(1).
    """

    #: Callable applied to each raw line to produce the item returned by ``__getitem__``.
    _parse: Callable[[str], Any] | None = None

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        cache_size: int = DEFAULT_CACHE_SIZE,
        encoding: str = "utf-8",
        strip: bool = True,
        skip_empty: bool = True,
        index_all: bool = False,
    ) -> None:
        _require_zstd()

        p = Path(path)
        if p.is_dir():
            self.paths: list[Path] = sorted(p.glob("*.zst"))
            if not self.paths:
                raise FileNotFoundError(f"no *.zst files in directory: {p}")
        else:
            if not p.is_file():
                raise FileNotFoundError(f"dataset file not found: {p}")
            self.paths = [p]

        self.chunk_size = chunk_size
        self.cache_size = cache_size
        self.encoding = encoding
        self.strip = strip
        self.skip_empty = skip_empty

        self._cache = LRUCache(cache_size)
        # One forward streaming cursor per file, so sequential reads advance
        # instead of re-decompressing from the start.
        self._cursors: dict[int, _Cursor] = {}
        # Cached per-chunk start records for O(log n) record -> chunk lookup.
        self._starts_cache: list[list[int] | None] = [None] * len(self.paths)
        self._lock = threading.Lock()

        # Per-file index: record ranges for each decompressed chunk.
        # _chunk_records[file_idx] = list of (start_record_in_file, num_records)
        self._chunk_records: list[list[tuple[int, int]]] = [[] for _ in self.paths]
        self._file_counts: list[int] = [0] * len(self.paths)
        self._closed = False

        if index_all:
            for i in range(len(self.paths)):
                self._index_file(i)

    # -- indexing ------------------------------------------------------------

    def _iter_decompressed_chunks(self, file_idx: int) -> Iterator[list[str]]:
        """Yield lists of complete lines, decompressing the stream incrementally.

        Zstd frames do not align with lines, so we drive a single streaming
        decompressor over the compressed file and split the resulting text on
        newlines, carrying any partial trailing line over to the next read.
        Each yielded list contains only complete lines.
        """
        dctx = zstd.ZstdDecompressor()
        carry = ""
        with open(self.paths[file_idx], "rb") as fh:
            dobj = dctx.decompressobj()
            while True:
                raw = fh.read(self.chunk_size)
                if not raw:
                    break
                carry += dobj.decompress(raw).decode(self.encoding, errors="replace")
                *complete, carry = carry.split("\n")
                if complete:
                    # Filter here so chunk contents always match the index,
                    # which counts only kept records.
                    yield [ln for ln in complete if self._keep(ln)]
            # Flush whatever the decompressor still holds, then any remainder.
            tail = dobj.flush() if hasattr(dobj, "flush") else b""
            carry += tail.decode(self.encoding, errors="replace")
            carry = carry.removesuffix("\n")
            complete = carry.split("\n") if carry else []
            if complete:
                yield [ln for ln in complete if self._keep(ln)]

    def _keep(self, line: str) -> bool:
        return bool(line.strip()) if self.skip_empty else True

    def _index_file(self, file_idx: int) -> None:
        """Fully index one file (all chunks)."""
        if self._chunk_records[file_idx]:
            return
        total = 0
        records: list[tuple[int, int]] = []
        for lines in self._iter_decompressed_chunks(file_idx):
            n = sum(1 for ln in lines if self._keep(ln))
            records.append((total, n))
            total += n
        self._chunk_records[file_idx] = records
        self._file_counts[file_idx] = total

    def _ensure_indexed(self) -> None:
        with self._lock:
            for i in range(len(self.paths)):
                if not self._chunk_records[i]:
                    self._index_file(i)

    @property
    def num_files(self) -> int:
        return len(self.paths)

    # -- sequence protocol ---------------------------------------------------

    def __len__(self) -> int:
        self._ensure_indexed()
        return sum(self._file_counts)

    def _locate(self, index: int) -> tuple[int, int]:
        """Map a global index to ``(file_idx, record_within_file)``."""
        self._ensure_indexed()
        if index < 0:
            index += sum(self._file_counts)
        if index < 0:
            raise IndexError("dataset index out of range")
        for i, count in enumerate(self._file_counts):
            if index < count:
                return i, index
            index -= count
        raise IndexError("dataset index out of range")

    def _chunk_for_record(self, file_idx: int, record: int) -> int:
        """Return the chunk id containing ``record`` within ``file_idx``.

        Chunk start records are non-decreasing, so this is a binary search
        rather than a linear scan (the index can hold thousands of chunks).
        """
        ranges = self._chunk_records[file_idx]
        starts = self._chunk_starts(file_idx)
        if not starts:
            raise IndexError(f"no chunks indexed for {self.paths[file_idx]}")
        cid = bisect.bisect_right(starts, record) - 1
        if cid >= 0:
            start, n = ranges[cid]
            if start <= record < start + n:
                return cid
        # Chunks holding only blank lines have zero records, so `starts`
        # repeats; walk backwards to the chunk that actually covers `record`.
        for back in range(cid, -1, -1):
            s2, n2 = ranges[back]
            if s2 <= record < s2 + n2:
                return back
        raise IndexError(
            f"record {record} not covered by index for {self.paths[file_idx]}"
        )

    def _chunk_starts(self, file_idx: int) -> list[int]:
        """Cached list of per-chunk start records, used for binary search."""
        starts = self._starts_cache[file_idx]
        if starts is None:
            starts = [s for s, _ in self._chunk_records[file_idx]]
            self._starts_cache[file_idx] = starts
        return starts

    def _get_chunk(self, file_idx: int, chunk_id: int) -> _Chunk:
        """Return a line-aligned chunk, using a forward cursor for locality.

        Zstd frames cannot be entered at arbitrary compressed offsets, so a chunk
        is only reachable by decompressing from the start of the file. Restarting
        from byte 0 on every cache miss makes sequential access O(n^2) -- each
        chunk re-decompresses everything before it.

        Instead we keep one *forward cursor* per file: a live streaming
        decompressor positioned just after chunk ``next_id``. When the requested
        chunk is at or ahead of the cursor -- the common case, since training
        reads records in order -- we simply continue forward and never re-read
        what we already passed. Only a request *behind* the cursor restarts from
        the beginning, which is the rare random-access path already served by
        the LRU in most cases.
        """
        key = (file_idx, chunk_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        ranges = self._chunk_records[file_idx]
        if not 0 <= chunk_id < len(ranges):
            raise IndexError(f"chunk {chunk_id} beyond end of {self.paths[file_idx]}")

        cursor = self._cursors.get(file_idx)
        if cursor is None or cursor.next_id > chunk_id:
            cursor = self._new_cursor(file_idx)

        while cursor.next_id <= chunk_id:
            try:
                lines = next(cursor.gen)
            except StopIteration as exc:  # pragma: no cover - defensive
                raise IndexError(
                    f"chunk {chunk_id} beyond end of {self.paths[file_idx]}"
                ) from exc
            cid = cursor.next_id
            start, n = ranges[cid]
            # `lines` is already filtered to the kept records by the chunk
            # iterator, so store it as-is and never re-filter per access.
            chunk = _Chunk(text=lines, start_record=start, num_records=n)
            self._cache.put((file_idx, cid), chunk)
            cursor.next_id += 1
            if cid == chunk_id:
                return chunk

        raise IndexError(f"chunk {chunk_id} beyond end of {self.paths[file_idx]}")

    def _new_cursor(self, file_idx: int) -> _Cursor:
        cursor = _Cursor(gen=self._iter_decompressed_chunks(file_idx), next_id=0)
        self._cursors[file_idx] = cursor
        return cursor

    def _raw_line(self, index: int) -> str:
        file_idx, record = self._locate(index)
        chunk_id = self._chunk_for_record(file_idx, record)
        chunk = self._get_chunk(file_idx, chunk_id)
        local = record - chunk.start_record

        # _Chunk.text already contains only kept records, so index straight in.
        lines = chunk.text
        if not 0 <= local < len(lines):
            raise IndexError(f"record {index} out of range within chunk")
        line = lines[local]
        return line.strip() if self.strip else line

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        raw = self._raw_line(index)
        if self._parse is not None:
            return self._parse(raw)
        return raw

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        try:
            n: Any = len(self)
        except (OSError, ValueError):  # pragma: no cover - defensive
            n = "?"
        return f"{type(self).__name__}(paths={len(self.paths)}, records={n})"

    # -- helpers -------------------------------------------------------------

    def iter_raw_lines(self) -> Iterator[str]:
        """Stream raw lines without maintaining an index (fastest sequential path).

        Use this when iterating the whole file once; it decompresses the stream
        front to back and never rewinds.
        """
        for file_idx in range(len(self.paths)):
            for lines in self._iter_decompressed_chunks(file_idx):
                for line in lines:
                    if self.skip_empty and not line.strip():
                        continue
                    yield line.strip() if self.strip else line

    @property
    def cache(self) -> LRUCache:
        """Expose the LRU for metrics/inspection."""
        return self._cache

    def close(self) -> None:
        self._closed = True
        self._cache.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ZstCsvDataset(ZstLineDataset):
    """CSV records stored one row per line inside a ``.zst`` file.

    Args:
        path: zstd-compressed CSV.
        fields: Optional column names. When omitted, the first record is used
            as the header. Pass an empty list to get positional tuples.
        delimiter: CSV delimiter.
        as_dict: Return ``dict`` rows (True) or ``tuple`` rows (False).

    Example:
        >>> ds = ZstCsvDataset("data.zst")      # header row inside the file
        >>> ds[0]["text"]
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fields: Sequence[str] | None = None,
        delimiter: str = ",",
        as_dict: bool = True,
        **kwargs: Any,
    ) -> None:
        self.delimiter = delimiter
        self.as_dict = as_dict
        # `has_header` is True when the file itself carries the header row.
        # Passing `fields` explicitly means the file has no header line.
        self.has_header = fields is None
        self._fields: list[str] | None = list(fields) if fields is not None else None
        super().__init__(path, **kwargs)
        if self.has_header:
            self._read_header()

    def _read_header(self) -> None:
        """Take record 0 as the column names; it is then hidden from indexing."""
        if self._fields is not None:
            return
        # The index must exist before we can address record 0.
        self._ensure_indexed()
        if sum(self._file_counts) == 0:
            raise ValueError(f"CSV dataset is empty: {self.paths}")
        header = super()._raw_line(0)
        self._fields = [f.strip() for f in next(
            csv.reader(io.StringIO(header), delimiter=self.delimiter)
        )]

    def _raw_line(self, index: int) -> str:
        # User-facing index i maps to stored record i+1 when a header exists.
        return super()._raw_line(index + 1 if self.has_header else index)

    def __len__(self) -> int:
        n = super().__len__()
        if not self.has_header:
            return n
        return max(n - 1, 0)

    def _row(self, line: str) -> Any:
        row = next(csv.reader(io.StringIO(line), delimiter=self.delimiter))
        if not self.as_dict:
            return tuple(row)
        fields = self._fields or []
        if len(row) < len(fields):
            row = row + [""] * (len(fields) - len(row))
        return dict(zip(fields, row))

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        # Normalize negative indices against the header-adjusted length so that
        # [-1] means the last data row, not the hidden header.
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("CSV index out of range")
        return self._row(self._raw_line(index))

    @property
    def fields(self) -> list[str] | None:
        return list(self._fields) if self._fields else None


class ZstJsonLinesDataset(ZstLineDataset):
    """One JSON object per line inside a ``.zst`` file (JSONL).

    Args:
        path: zstd-compressed JSONL.
        key: Optional key to extract from each object.
        lines: When True, tolerate invalid JSON by returning the raw string.

    Example:
        >>> ds = ZstJsonLinesDataset("corpus.zst", key="text")
        >>> ds[0]
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        key: str | None = None,
        lines: bool = False,
        **kwargs: Any,
    ) -> None:
        self.key = key
        self.lines = lines
        super().__init__(path, **kwargs)

    def _obj(self, line: str) -> Any:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if self.lines:
                return line
            raise
        if self.key is not None:
            return obj.get(self.key) if isinstance(obj, dict) else None
        return obj

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self._obj(self._raw_line(index))
