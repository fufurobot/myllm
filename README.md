# myllm

A Training framework for my own llm experiments

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/fufurobot/myllm)

## Startup

### Local setup

This project uses [uv](https://docs.astral.sh/uv/). Pick **exactly one** torch
backend — the `cpu` and `cu118` extras are mutually exclusive and provide the
same packages from different indexes.

```bash
# GPU machine with an NVIDIA driver supporting CUDA 11.8
uv sync --extra cu118

# CPU-only machine
uv sync --extra cpu
```

Both backends are driven by a single `uv.lock`, so switching between them only
requires re-running `uv sync` with the other extra.

### Environment variables

Copy the example file and adjust it for your machine:

```bash
cp .env.example .env
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `HF_ENDPOINT` | Hugging Face mirror used by `download-hf.sh` / `hfd.sh` | `https://hf-mirror.com` |
| `HFD_EXTRA_ARGS` | Extra flags forwarded to `hfd.sh` (e.g. `--hf_token <token>`) | *(empty)* |
| `HF_TOKEN` | Hugging Face access token for gated/private repos | *(empty)* |
| `DRY_RUN` | Set to `true` to print download commands without running them | `false` |

### Downloading models and datasets

`download-hf.sh` scans for `.gitkeep` markers under `*/models/<user>/<repo>/`
and `*/datasets/<user>/<repo>/` and fetches each repo through `hfd.sh`:

```bash
./download-hf.sh
```

Preview the commands without downloading anything:

```bash
DRY_RUN=true ./download-hf.sh
```

### GitHub Codespaces

Clicking the badge above creates a Codespace that provisions itself:

1. Installs `uv`.
2. Runs `uv sync --extra cpu` (Codespaces runners have no GPU).
3. Downloads models and datasets declared by `.gitkeep` markers via
   `download-hf.sh`.

To use a CUDA 11.8 build inside the Codespace, run:

```bash
uv sync --extra cu118
```

## Model: Linear RNN

The model is a pure linear recurrent network evaluated with a parallel prefix
scan. Per layer, with `x_t` the input at step `t`:

```
h_t = W_x @ x_t + W_h @ h_(t-1) + b        (h_0 = 0)
y_t = h_t
```

This is the classic Elman recurrence with a linear activation. `W_h` is the
only path through time; `W_x` and `b` act pointwise per step.

### Row-stochastic transition matrices

Backpropagation through time multiplies `W_h` once per step, so the gradient
scales roughly like `W_h^T`. If the spectral radius exceeds 1 the gradient
grows exponentially; below 1 it vanishes.

`W_h` is therefore **row-stochastic**: every row is normalised to sum to 1 at
use time (the raw weights stay unconstrained so gradients flow normally).
A row-stochastic matrix has spectral radius exactly 1 — the all-ones vector is
a right eigenvector with eigenvalue 1 — and row-stochastic matrices are closed
under multiplication:

```
if A1 = 1 and B1 = 1, then (AB)1 = A(B1) = A1 = 1
```

So however many steps the recurrence takes, the operator stays bounded and the
gradient cannot blow up. Measured on a 16-dim layer, summing gradients over a
sequence of length `T`:

| `T` | row-stochastic | unconstrained (spectral radius > 1) |
| --- | --- | --- |
| 10 | 3.7e+02 | 8.9e+02 |
| 50 | 3.2e+03 | 9.9e+09 |
| 200 | 3.4e+04 | `inf` |
| 400 | 4.8e+04 | `nan` |

Row-stochastic growth is linear in `T`; the unconstrained version overflows.

**Numerical safety.** Dividing by a raw row sum is unsafe when that sum is near
zero (division explodes) or enormous (overflow). The normaliser first *scales*
each row — which preserves its direction — into `[min_row_sum, max_row_sum]`,
then divides by the true sum. All-zero rows fall back to a uniform
distribution. Rows scaled to `1e-12` and `1e20` both normalise to exactly 1.

### Parallel prefix scan

The linear recurrence is an affine scan, so it need not be evaluated with a
`T`-step loop. `prefix_scan` composes affine maps `(A, c)` representing
`h -> A h + c` with a Blelloch-style doubling scan, giving log-depth parallel
evaluation:

```
(A_2, c_2) o (A_1, c_1) = (A_2 A_1, A_2 c_1 + c_2)
```

This composition is associative, so the scan is exact (verified equal to the
step loop for `T` in 1, 2, 3, 5, 8, 17, 64 to ~1e-6). It is built entirely
out-of-place so autograd can differentiate through it, and it stays finite at
`T = 4096`. Row-stochasticity is what makes the scan numerically safe: the
composed operators cannot grow.

### Frozen, factorised embeddings

`W_in` (`[d, V]`) and `W_out` (`[V, d]`) are initialised from a factorisation of
the bigram count matrix and are **frozen by default** — only the recurrent
layers train.

The rationale: if the recurrent path is a pure pass-through, the network
reduces to `softmax(W_out @ W_in[:, x])`, a bigram predictor. Making that the
model's *default* behaviour means `W_in` and `W_out` should factorise the
transition matrix — an `N x N` matrix decomposed into `N x d` (input
embeddings) and `d x N` (output embeddings). This is also why the two matrices
are **never tied**: they are the two factors, and tying them would force a
symmetric transition matrix.

Two factorisation methods are available via `--init-method`:

| Method | Operates on | Objective |
| --- | --- | --- |
| `svd` | `ln P(next \| prev)` | Squared error on the log-probability scale; rare and frequent transitions weighted comparably |
| `nmf` | Raw counts | KL divergence (multiplicative updates); both factors stay non-negative |

### Language-aware tokenization

Tokenization mode is chosen from the eBible `translations.csv` `script` column,
falling back to a space-ratio heuristic:

- **Word mode** — languages that use the space as a delimiter. Text is
  lowercased and split on whitespace.
- **Character mode** — languages that do not (CJK, Thai, Khmer, ...). Text is
  tokenized per Unicode codepoint.

## Data: lazy `.zst` datasets

Training data is read from Zstandard-compressed files where **one record per
line** is the storage unit. `myllm.dataset` provides a unified lazy loader:

| Class | Line format |
| --- | --- |
| `ZstLineDataset` | base class; each line is one raw record |
| `ZstCsvDataset` | CSV rows (`dict` or `tuple`); header row hidden from indices |
| `ZstJsonLinesDataset` | one JSON object per line; optional key extraction |

All three inherit `ZstLineDataset`, so they share the same lazy behaviour:

- **Lazy, indexed.** A growing index maps record -> chunk by decompressing
  the stream once in `chunk_size` compressed-byte steps. The file is never
  fully decompressed into memory.
- **LRU chunk cache.** Decompressed chunks are held in a bounded LRU
  (`cache_size`, default 32), so sequential scans decode each chunk once and
  repeated access is cheap.
- **Forward cursor.** A request at or ahead of the last decode continues
  forward instead of restarting from byte 0. This is what keeps sequential
  access linear — with per-access work proportional to chunk size, reading N
  records took 50 s instead of 0.045 s.
- **Binary search.** Per-chunk start records are `bisect`-searched.

Inspect any file from the CLI (this path needs only `zstandard`, not torch):

```bash
myllm dataset notebooks/bible_sentences.csv.zst --num 3
myllm dataset data.jsonl.zst --format jsonl --key text
myllm dataset corpus/ --format lines --cache-size 64
```

The format is auto-detected from the filename (`.csv.zst`, `.jsonl.zst`,
otherwise `lines`).

Reading `.zst` files requires the `zstandard` package, which is declared in
`pyproject.toml`; a clear `ZstdUnavailableError` is raised if it is missing.

## Training

The eBible corpus stores one verse per line, so each non-empty line is one
training example.

```bash
# English (word mode), SVD init
uv run myllm train --language eng --dim 128 --num-layers 2 --max-steps 500

# Chinese (character mode), NMF init
uv run myllm train --language cmn --init-method nmf --dim 128 --max-steps 500
```

Inspect a language before training to check its detected mode:

```bash
uv run myllm inspect --language jpn
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--language` | eBible language code (`eng`, `cmn`, `jpn`, ...) |
| `--init-method` | `svd` (default) or `nmf` |
| `--dim` / `--num-layers` | Model width and depth |
| `--freeze-embeddings` | On by default; `--unfreeze-embeddings` trains `W_in`/`W_out` |
| `--max-files` | Limit corpus files (useful for quick runs) |
| `--max-vocab` | Cap the vocabulary size |
| `--device` | `auto` (default), `cpu`, `cuda`, `mps` |

Artifacts are written to `runs/token_shift/`: `model.pt`, `tokenizer.json`,
`config.json`, and `metrics.jsonl`.

### CUDA container

The CUDA image is published to GHCR on each push to `main`. It uses PyTorch
2.8.0 with CUDA 12.8 and checks that the build includes `sm_120`. The eBible
corpus is intentionally mounted at runtime rather than copied into the image.

```bash
docker run --rm --gpus all \
  -v "$PWD/repo/dataset/ebible:/app/repo/dataset/ebible:ro" \
  -v "$PWD/runs:/app/runs" \
  ghcr.io/fufurobot/myllm:main \
  train --language eng --dim 128 --num-layers 20 --max-steps 10000 --device cuda
```

## Development

```bash
uv run pytest
uv run ruff check .
```
