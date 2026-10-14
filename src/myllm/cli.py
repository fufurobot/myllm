"""Command-line interface for myllm."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_METADATA_CSV,
    load_corpus,
)
from .trainer import TrainConfig, train


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

    return p


def _cmd_train(args: argparse.Namespace) -> int:
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
        seed=args.seed,
        out_dir=args.out_dir,
        device=args.device,
        log_interval=args.log_interval,
    )
    train(cfg)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    corpus = load_corpus(
        args.language, args.corpus_dir, args.metadata_csv, max_files=args.max_files
    )
    from .data import build_tokenizer

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
