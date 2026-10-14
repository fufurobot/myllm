"""Minimal trainer for :class:`~myllm.model.TokenShiftLM`.

Pipeline
--------
1. Load verse lines for a language from the eBible corpus.
2. Build a language-aware tokenizer (word mode for space-delimited languages,
   character mode otherwise).
3. Accumulate bigram counts and factorise them (SVD or NMF) to initialise the
   frozen input embedding and output linear.
4. Train the recurrent layers with next-token cross-entropy, backpropagating
   through time over each verse line.

``accelerate`` handles device placement and (optionally) mixed precision.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import Corpus, build_tokenizer, load_corpus, split_lines
from .init import (
    build_count_matrix,
    factorize_nmf,
    factorize_svd,
    normalize_factors,
)
from .model import ModelConfig, TokenShiftLM
from .tokenizer import Tokenizer


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    language: str = "eng"
    corpus_dir: Path = Path("repo/dataset/ebible/corpus")
    metadata_csv: Path = Path("repo/dataset/ebible/metadata/translations.csv")

    dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0
    freeze_embeddings: bool = True

    init_method: str = "svd"  # "svd" or "nmf"
    nmf_iters: int = 200
    svd_add_k: float = 1.0

    min_freq: int = 2
    max_vocab: int | None = None

    seq_len: int = 64
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 500
    eval_interval: int = 50
    eval_batches: int = 20
    val_fraction: float = 0.05
    warmup_steps: int = 20

    max_files: int | None = None
    seed: int = 0
    out_dir: Path = Path("runs/token_shift")
    device: str = "auto"
    log_interval: int = 10


class LineDataset(Dataset):
    """Fixed-length windows over encoded verse lines.

    Lines are concatenated with ``<eos>`` separators and then chopped into
    windows of ``seq_len + 1`` tokens, giving ``seq_len`` next-token prediction
    targets per window. This keeps batches tensor-shaped without padding.
    """

    def __init__(self, encoded: list[list[int]], seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        stream: list[int] = []
        for ids in encoded:
            stream.extend(ids)
        self.seq_len = seq_len
        self.stream = torch.tensor(stream, dtype=torch.long)
        # Each sample needs seq_len inputs + 1 target.
        self.n_samples = max(0, (len(self.stream) - 1) // seq_len)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.stream[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def initialize_embeddings(
    model: TokenShiftLM,
    encoded_train: list[list[int]],
    tokenizer: Tokenizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Factorise the bigram count matrix and install the frozen embeddings."""
    counts = build_count_matrix(encoded_train, vocab_size=len(tokenizer))

    if cfg.init_method == "svd":
        w_in, w_out = factorize_svd(counts, dim=cfg.dim, add_k=cfg.svd_add_k)
    elif cfg.init_method == "nmf":
        w_in, w_out = factorize_nmf(counts, dim=cfg.dim, n_iter=cfg.nmf_iters, seed=cfg.seed)
    else:
        raise ValueError(f"init_method must be 'svd' or 'nmf', got {cfg.init_method!r}")

    w_in, w_out = normalize_factors(w_in, w_out)
    model.load_factorized_embeddings(w_in, w_out)

    recon = w_out @ w_in
    target = counts.log_prob(add_k=cfg.svd_add_k)
    import numpy as np

    finite = np.isfinite(target)
    corr = float(
        np.corrcoef(recon[finite].ravel(), target[finite].ravel())[0, 1]
    ) if finite.sum() > 1 else float("nan")

    return {
        "bigram_total": counts.total,
        "factor_recon_corr": corr,
        "w_in_norm": float(np.linalg.norm(w_in)),
        "w_out_norm": float(np.linalg.norm(w_out)),
    }


@torch.no_grad()
def evaluate(
    model: TokenShiftLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    """Average cross-entropy / perplexity over a limited number of batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"
        )
        total_loss += float(loss)
        total_tokens += y.numel()

    model.train()
    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "tokens": 0}
    mean = total_loss / total_tokens
    return {"loss": mean, "ppl": math.exp(min(mean, 20.0)), "tokens": total_tokens}


def train(cfg: TrainConfig) -> dict:
    """Run the full pipeline and return a summary dict.

    Artifacts written to ``cfg.out_dir``: ``model.pt``, ``tokenizer.json``,
    ``config.json``, ``metrics.jsonl``.
    """
    torch.manual_seed(cfg.seed)

    device = _resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. data -----------------------------------------------------------
    corpus: Corpus = load_corpus(
        cfg.language, cfg.corpus_dir, cfg.metadata_csv, max_files=cfg.max_files
    )
    tokenizer = build_tokenizer(corpus, min_freq=cfg.min_freq, max_vocab=cfg.max_vocab)

    train_lines, val_lines = split_lines(corpus.lines, cfg.val_fraction, seed=cfg.seed)
    encoded_train = tokenizer.encode_many(train_lines, add_eos=True)
    encoded_val = tokenizer.encode_many(val_lines, add_eos=True) if val_lines else []

    print(
        f"[data] language={corpus.language_code} mode={corpus.mode} "
        f"script={corpus.profile.script!r} lines={len(corpus.lines)} "
        f"vocab={len(tokenizer)}"
    )

    train_ds = LineDataset(encoded_train, cfg.seq_len)
    val_ds = LineDataset(encoded_val, cfg.seq_len) if encoded_val else None

    # --- 2. model ----------------------------------------------------------
    model_cfg = ModelConfig(
        vocab_size=len(tokenizer),
        dim=cfg.dim,
        num_layers=cfg.num_layers,
        freeze_embeddings=cfg.freeze_embeddings,
        dropout=cfg.dropout,
    )
    model = TokenShiftLM(model_cfg).to(device)

    # --- 3. factorised init ------------------------------------------------
    init_stats = initialize_embeddings(model, encoded_train, tokenizer, cfg)
    params = model.parameter_summary()
    print(
        f"[init] method={cfg.init_method} recon_corr={init_stats['factor_recon_corr']:.3f} "
        f"params total={params['total']:,} trainable={params['trainable']:,} "
        f"frozen={params['frozen']:,}"
    )

    # --- 4. optimiser ------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters: check freeze_embeddings/num_layers")
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "not enough data for one window; lower seq_len or use a larger corpus"
        )

    metrics_path = out_dir / "metrics.jsonl"
    history: list[dict] = []
    model.train()
    start = time.time()
    step = 0
    running_loss = 0.0
    running_count = 0

    data_iter = iter(train_loader)
    while step < cfg.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        lr = _lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()

        running_loss += float(loss)
        running_count += 1
        step += 1

        if step % cfg.log_interval == 0 or step == 1:
            avg = running_loss / max(running_count, 1)
            elapsed = time.time() - start
            print(
                f"[train] step {step}/{cfg.max_steps} loss={avg:.4f} "
                f"ppl={math.exp(min(avg, 20.0)):.2f} lr={lr:.2e} {elapsed:.1f}s"
            )
            running_loss = 0.0
            running_count = 0

        if val_loader is not None and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
            val = evaluate(model, val_loader, device, cfg.eval_batches)
            record = {"step": step, "lr": lr, "val_loss": val["loss"], "val_ppl": val["ppl"]}
            history.append(record)
            print(f"[eval] step {step} val_loss={val['loss']:.4f} val_ppl={val['ppl']:.2f}")
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    # --- 5. artifacts ------------------------------------------------------
    tokenizer.save(out_dir / "tokenizer.json")
    torch.save(
        {"model_state": model.state_dict(), "model_config": asdict(model_cfg)},
        out_dir / "model.pt",
    )
    summary = {
        "language": corpus.language_code,
        "mode": corpus.mode,
        "script": corpus.profile.script,
        "vocab_size": len(tokenizer),
        "params": params,
        "init": init_stats,
        "steps": step,
        "history": history,
        "out_dir": str(out_dir),
        "seconds": time.time() - start,
    }
    (out_dir / "config.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"[done] artifacts written to {out_dir}")
    return summary
