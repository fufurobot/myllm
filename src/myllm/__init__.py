"""myllm - an experimental language modelling framework.

Heavy submodules (which need numpy/torch) are imported lazily so that lightweight
pieces such as :mod:`myllm.dataset` can be used without the full ML stack.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

#: Public name -> submodule that defines it. Resolved on first access.
_LAZY: dict[str, str] = {
    "Corpus": "data",
    "build_tokenizer": "data",
    "load_corpus": "data",
    "split_lines": "data",
    "CountMatrix": "init",
    "build_count_matrix": "init",
    "factorize_nmf": "init",
    "factorize_svd": "init",
    "normalize_factors": "init",
    "ModelConfig": "model",
    "TokenShiftLayer": "model",
    "TokenShiftLM": "model",
    "LinearRNN": "linear_rnn",
    "LinearRNNConfig": "linear_rnn",
    "RowStochastic": "linear_rnn",
    "prefix_scan": "linear_rnn",
    "Tokenizer": "tokenizer",
    "load_language_profiles": "tokenizer",
    "TrainConfig": "trainer",
    "evaluate": "trainer",
    "train": "trainer",
    "ZstLineDataset": "dataset",
    "ZstCsvDataset": "dataset",
    "ZstJsonLinesDataset": "dataset",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    """Resolve lazily-exported names (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))


def main() -> None:
    """Console-script entry point."""
    from .cli import main as cli_main

    raise SystemExit(cli_main())
