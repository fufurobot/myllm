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
