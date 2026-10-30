"""eBible corpus loading.

The corpus is one file per translation (``<lang>-<project>.txt``) with one verse
per line. Blank lines mark verses absent from that translation.  Sentences are
already newline-separated, so each non-empty line is treated as one training
example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tokenizer import (
    LanguageProfile,
    Tokenizer,
    infer_profile_from_text,
    load_language_profiles,
)

DEFAULT_CORPUS_DIR = Path("repo/dataset/ebible/corpus")
DEFAULT_METADATA_CSV = Path("repo/dataset/ebible/metadata/translations.csv")

# Lines that are placeholders rather than text.
SKIP_LINE_VALUES = {"<range>"}


@dataclass
class Corpus:
    """Loaded verse lines for one language."""

    language_code: str
    profile: LanguageProfile
    lines: list[str]
    source_files: list[Path]

    def __len__(self) -> int:
        return len(self.lines)

    @property
    def mode(self) -> str:
        return self.profile.mode


def find_corpus_files(
    language_code: str,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
) -> list[Path]:
    """Return all corpus files for ``language_code`` (e.g. ``eng``)."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    return sorted(corpus_dir.glob(f"{language_code}-*.txt"))


def read_verse_lines(path: Path) -> list[str]:
    """Read one corpus file into a list of non-empty verse lines."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in SKIP_LINE_VALUES:
            continue
        out.append(line)
    return out


def load_corpus(
    language_code: str,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    metadata_csv: Path = DEFAULT_METADATA_CSV,
    max_files: int | None = None,
) -> Corpus:
    """Load verse lines for a language and resolve its tokenization profile.

    Raises:
        FileNotFoundError: if no corpus file matches the language code.
    """
    files = find_corpus_files(language_code, corpus_dir)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"no corpus files matching {language_code}-*.txt in {corpus_dir}"
        )

    lines: list[str] = []
    for path in files:
        lines.extend(read_verse_lines(path))

    if not lines:
        # Some translations ship as placeholder files (all newlines, no text);
        # catching this here gives a clearer error than failing later.
        raise ValueError(
            f"corpus files for {language_code!r} contain no text: "
            f"{', '.join(p.name for p in files)}"
        )

    profiles = load_language_profiles(metadata_csv)
    profile = profiles.get(language_code)
    if profile is None:
        # Not in the metadata: fall back to inspecting the text itself.
        profile = infer_profile_from_text("\n".join(lines[:5000]), language_code)

    return Corpus(language_code, profile, lines, files)


def build_tokenizer(
    corpus: Corpus,
    min_freq: int = 2,
    max_vocab: int | None = None,
) -> Tokenizer:
    """Train a tokenizer whose mode matches the corpus language profile."""
    return Tokenizer.train(
        corpus.lines,
        mode=corpus.mode,
        min_freq=min_freq,
        max_vocab=max_vocab,
        language_code=corpus.language_code,
    )


def split_lines(
    lines: list[str],
    val_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split lines into (train, val).

    The split is contiguous in original order rather than shuffled: verses are
    author-ordered, so a random split would leak adjacent context between the
    two sets and overstate validation quality.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    n_val = int(len(lines) * val_fraction)
    if n_val == 0:
        return lines, []
    return lines[:-n_val], lines[-n_val:]


# ---------------------------------------------------------------------------
# Zst-backed corpus loading
# ---------------------------------------------------------------------------


def load_zst_corpus(
    path: Path,
    text_field: str | None = "sentence",
    language_code: str | None = None,
    metadata_csv: Path = DEFAULT_METADATA_CSV,
    limit: int | None = None,
    cache_size: int = 32,
    chunk_size: int = 1 << 20,
) -> Corpus:
    """Load verse lines from a ``.zst`` dataset.

    Supports the three line formats handled by :mod:`myllm.dataset`:

    * ``*.csv.zst`` -- read with :class:`~myllm.dataset.ZstCsvDataset`; the
      ``text_field`` column supplies the sentence. This is the format of the
      released ``bible_sentences.csv.zst`` artifact.
    * ``*.jsonl.zst`` -- read with
      :class:`~myllm.dataset.ZstJsonLinesDataset`; ``text_field`` is the key.
    * anything else -- read with :class:`~myllm.dataset.ZstLineDataset`, one
      record per line (``text_field`` is ignored).

    Args:
        path: ``.zst`` file (or directory of them).
        text_field: Column/key holding the sentence for CSV/JSONL inputs.
        language_code: Language code used for tokenization mode. When omitted
            it is taken from a ``lang`` column if present, else guessed from
            the text.
        metadata_csv: eBible metadata used to resolve the tokenization script.
        limit: Stop after this many lines (useful for smoke tests).
        cache_size: LRU chunk cache size passed to the loader.
        chunk_size: Compressed bytes per read.

    Raises:
        KeyError: if ``text_field`` is absent from the CSV/JSONL records.
    """
    from .dataset import ZstCsvDataset, ZstJsonLinesDataset, ZstLineDataset

    name = path.name.lower()
    common = {"cache_size": cache_size, "chunk_size": chunk_size}

    lines: list[str] = []
    detected_lang: str | None = None

    if name.endswith(".csv.zst"):
        ds = ZstCsvDataset(path, **common)
        fields = ds.fields or []
        if text_field is not None and text_field not in fields:
            raise KeyError(
                f"text_field {text_field!r} not in CSV fields {fields}"
            )
        for i in range(len(ds)):
            row = ds[i]
            if text_field is not None:
                val = (row.get(text_field) or "").strip()
                if not val or val in SKIP_LINE_VALUES:
                    continue
                lines.append(val)
            else:
                lines.append(",".join(str(v) for v in row))
            if detected_lang is None and "lang" in row:
                detected_lang = (row.get("lang") or "").strip() or None
            if limit is not None and len(lines) >= limit:
                break
    elif name.endswith((".jsonl.zst", ".json.zst", ".ndjson.zst")):
        ds = ZstJsonLinesDataset(path, key=text_field, **common)
        for i in range(len(ds)):
            val = ds[i]
            if text_field is not None and val is None:
                raise KeyError(f"key {text_field!r} missing from a JSONL record")
            if isinstance(val, str):
                val = val.strip()
                if not val or val in SKIP_LINE_VALUES:
                    continue
                lines.append(val)
            if limit is not None and len(lines) >= limit:
                break
    else:
        ds = ZstLineDataset(path, **common)
        for i in range(len(ds)):
            val = ds[i]
            if isinstance(val, str) and val.strip():
                lines.append(val.strip())
            if limit is not None and len(lines) >= limit:
                break

    if not lines:
        raise ValueError(f"no text records found in {path}")

    code = language_code or detected_lang or "??"
    profiles = load_language_profiles(metadata_csv)
    profile = profiles.get(code)
    if profile is None:
        profile = infer_profile_from_text("\n".join(lines[:5000]), code)

    source_files = getattr(ds, "paths", [Path(path)])
    return Corpus(code, profile, lines, list(source_files))
