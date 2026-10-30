"""Command-line interface for myllm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Imported at module scope would pull in numpy/torch for every command, which
# defeats the point of the lazy package. Only the dataset/corpus paths are
# needed for `dataset` and `inspect`, so those are imported where used.
from .dataset import (
    ZstCsvDataset,
    ZstdUnavailableError,
    ZstJsonLinesDataset,
    ZstLineDataset,
)

DEFAULT_CORPUS_DIR = Path("repo/dataset/ebible/corpus")
DEFAULT_METADATA_CSV = Path("repo/dataset/ebible/metadata/translations.csv")

_DATASET_BY_FORMAT = {
    "lines": ZstLineDataset,
    "csv": ZstCsvDataset,
    "jsonl": ZstJsonLinesDataset,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="myllm",
        description="Train a TokenShift RNN language model on the eBible corpus.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- train ---------------------------------------------------------------
    t = sub.add_parser("train", help="train a model on one language")
    t.add_argument("--language", default="eng", help="language code, e.g. eng, cmn, jpn")
    t.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    t.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)

    t.add_argument("--dim", type=int, default=128)
    t.add_argument("--num-layers", type=int, default=2)
    t.add_argument("--dropout", type=float, default=0.0)

    t.add_argument("--init-method", choices=["svd", "nmf"], default="svd")
    t.add_argument("--nmf-iters", type=int, default=200)
    t.add_argument("--svd-add-k", type=float, default=1.0)

    t.add_argument("--min-freq", type=int, default=2)
    t.add_argument("--max-vocab", type=int, default=None)

    t.add_argument("--seq-len", type=int, default=64)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--max-steps", type=int, default=500)
    t.add_argument("--eval-interval", type=int, default=50)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--val-fraction", type=float, default=0.05)
    t.add_argument("--warmup-steps", type=int, default=20)
    t.add_argument("--grad-clip", type=float, default=1.0)

    t.add_argument("--max-files", type=int, default=None)
    t.add_argument(
        "--zst",
        type=Path,
        default=None,
        help="train from a .zst artifact (csv/jsonl/lines) instead of the .txt corpus",
    )
    t.add_argument(
        "--zst-text-field",
        default="sentence",
        help="CSV column / JSONL key holding the sentence (default: sentence)",
    )
    t.add_argument("--zst-limit", type=int, default=None, help="max records to read")
    t.add_argument(
        "--no-row-stochastic",
        action="store_true",
        help="disable row normalization of the transition matrices",
    )
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out-dir", type=Path, default=Path("runs/token_shift"))
    t.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    t.add_argument("--unfreeze-embeddings", action="store_true")
    t.add_argument("--log-interval", type=int, default=10)
    t.set_defaults(func=_cmd_train)

    # -- inspect -------------------------------------------------------------
    i = sub.add_parser("inspect", help="show corpus/tokenizer stats for a language")
    i.add_argument("--language", default="eng")
    i.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    i.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    i.add_argument("--max-files", type=int, default=None)
    i.set_defaults(func=_cmd_inspect)

    # -- dataset -------------------------------------------------------------
    d = sub.add_parser(
        "dataset",
        help="inspect a line-separated .zst dataset (lines/csv/jsonl)",
    )
    d.add_argument("path", type=Path, help=".zst file or directory of .zst files")
    d.add_argument(
        "--format",
        choices=["auto", *sorted(_DATASET_BY_FORMAT)],
        default="auto",
        help="record format stored per line (default: lines)",
    )
    d.add_argument("--num", type=int, default=3, help="how many records to print")
    d.add_argument("--at", type=int, default=0, help="first record index to print")
    d.add_argument("--cache-size", type=int, default=32, help="LRU chunk cache entries")
    d.add_argument("--chunk-size", type=int, default=1 << 20, help="compressed chunk bytes")
    d.add_argument("--key", default=None, help="jsonl: extract this key from each object")
    d.add_argument("--delimiter", default=",", help="csv: field delimiter")
    d.add_argument("--fields", default=None, help="csv: comma-separated column names")
    d.add_argument("--no-strip", action="store_true", help="keep whitespace as-is")
    d.set_defaults(func=_cmd_dataset)

    # -- rnn-demo ------------------------------------------------------------
    r = sub.add_parser(
        "rnn-demo",
        help="run the LinearRNN reference forward pass and show stability stats",
    )
    r.add_argument("--input-size", type=int, default=3)
    r.add_argument("--hidden-size", type=int, default=4)
    r.add_argument("--seq-len", type=int, default=5)
    r.add_argument("--num-layers", type=int, default=1)
    r.add_argument(
        "--no-row-stochastic",
        action="store_true",
        help="disable row normalization (shows the gradient boom)",
    )
    r.set_defaults(func=_cmd_rnn_demo)

    return p


def _cmd_train(args: argparse.Namespace) -> int:
    from .trainer import TrainConfig, train

    cfg = TrainConfig(
        language=args.language,
        corpus_dir=args.corpus_dir,
        metadata_csv=args.metadata_csv,
        dim=args.dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_embeddings=not args.unfreeze_embeddings,
        init_method=args.init_method,
        nmf_iters=args.nmf_iters,
        svd_add_k=args.svd_add_k,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        val_fraction=args.val_fraction,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        max_files=args.max_files,
        zst_path=args.zst,
        zst_text_field=args.zst_text_field,
        zst_limit=args.zst_limit,
        row_stochastic=not args.no_row_stochastic,
        seed=args.seed,
        out_dir=args.out_dir,
        device=args.device,
        log_interval=args.log_interval,
    )
    train(cfg)
    return 0


def _detect_format(path: Path) -> str:
    """Guess the record format from the filename (does not read the file).

    `*.csv.zst` -> csv, `*.jsonl.zst` / `*.json.zst` -> jsonl, else lines.
    """
    name = path.name.lower()
    if name.endswith(".csv.zst"):
        return "csv"
    if name.endswith((".jsonl.zst", ".json.zst", ".ndjson.zst")):
        return "jsonl"
    return "lines"


def _cmd_dataset(args: argparse.Namespace) -> int:
    if args.format == "auto":
        args.format = _detect_format(args.path)
        print(f"detected format: {args.format}")
    cls = _DATASET_BY_FORMAT[args.format]
    kwargs: dict = {
        "cache_size": args.cache_size,
        "chunk_size": args.chunk_size,
        "strip": not args.no_strip,
    }
    if args.format == "jsonl" and args.key:
        kwargs["key"] = args.key
    if args.format == "csv":
        kwargs["delimiter"] = args.delimiter
        if args.fields:
            kwargs["fields"] = [f.strip() for f in args.fields.split(",")]

    try:
        ds = cls(args.path, **kwargs)
    except ZstdUnavailableError as exc:
        print(f"error: {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2

    n = len(ds)
    print(f"class       : {type(ds).__name__}")
    print(f"files       : {len(ds.paths)}")
    print(f"records     : {n:,}")
    if args.format == "csv":
        print(f"fields      : {ds.fields}")
    print(f"chunk size  : {args.chunk_size:,} compressed bytes")
    print(f"cache       : {args.cache_size} chunks (LRU)")

    print()
    for i in range(args.at, min(args.at + max(args.num, 0), n)):
        rec = ds[i]
        if isinstance(rec, dict):
            text = json.dumps(rec, ensure_ascii=False)
        else:
            text = repr(rec)
        print(f"[{i}] {text[:300]}")

    print()
    print(f"cache hits/misses: {ds.cache.hits}/{ds.cache.misses}")
    return 0


def _cmd_rnn_demo(args: argparse.Namespace) -> int:
    import torch

    from .linear_rnn import LinearRNN, LinearRNNConfig

    cfg = LinearRNNConfig(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        row_stochastic=not args.no_row_stochastic,
    )
    model = LinearRNN(cfg)
    x = torch.randn(args.seq_len, args.input_size)
    out = model(x)

    print(f"input        : {tuple(x.shape)}")
    print(f"output       : {tuple(out.shape)}")
    print(f"row_stochastic: {cfg.row_stochastic}")
    if cfg.row_stochastic:
        w = model.layers[0].w_h.effective().detach()
        print(f"row sums     : {w.sum(1)[:4].tolist()}")
        print(f"spectral radius: {model.spectral_radius_of_transition():.6f}")

    loss = out.sum()
    loss.backward()
    g = model.layers[0].w_x.weight.grad
    print(f"grad norm    : {float(g.norm()):.4e}")
    print(f"grad finite  : {bool(torch.isfinite(g).all())}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .data import build_tokenizer, load_corpus

    corpus = load_corpus(
        args.language, args.corpus_dir, args.metadata_csv, max_files=args.max_files
    )
    tok = build_tokenizer(corpus, min_freq=1)
    n_chars = sum(len(line) for line in corpus.lines)
    n_spaces = sum(line.count(" ") for line in corpus.lines)

    print(f"language      : {corpus.language_code}")
    print(f"script        : {corpus.profile.script}")
    print(f"tokenize mode : {corpus.mode}")
    print(f"files         : {len(corpus.source_files)}")
    for f in corpus.source_files[:5]:
        print(f"                {f}")
    print(f"verse lines   : {len(corpus.lines):,}")
    print(f"characters    : {n_chars:,}")
    print(f"space ratio   : {n_spaces / max(n_chars, 1):.4f}")
    print(f"vocab (minf=1): {len(tok):,}")

    sample = corpus.lines[0] if corpus.lines else ""
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    print(f"\nsample text   : {sample[:80]}")
    print(f"sample ids    : {ids[:20]}{'...' if len(ids) > 20 else ''}")
    print(f"decoded       : {tok.decode(ids)[:80]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
