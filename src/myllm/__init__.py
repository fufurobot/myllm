"""myllm - a TokenShift RNN language modelling framework."""

from .data import Corpus, build_tokenizer, load_corpus, split_lines
from .init import (
    CountMatrix,
    build_count_matrix,
    factorize_nmf,
    factorize_svd,
    normalize_factors,
)
from .model import ModelConfig, TokenShiftLayer, TokenShiftLM
from .tokenizer import Tokenizer, load_language_profiles
from .trainer import TrainConfig, evaluate, train

__version__ = "0.1.0"

__all__ = [
    "Corpus",
    "CountMatrix",
    "ModelConfig",
    "TokenShiftLM",
    "TokenShiftLayer",
    "Tokenizer",
    "TrainConfig",
    "build_count_matrix",
    "build_tokenizer",
    "evaluate",
    "factorize_nmf",
    "factorize_svd",
    "load_corpus",
    "load_language_profiles",
    "normalize_factors",
    "split_lines",
    "train",
]


def main() -> None:
    """Console-script entry point."""
    from .cli import main as cli_main

    raise SystemExit(cli_main())
