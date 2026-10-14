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

## Model: TokenShift RNN

A plain recurrent language model. Each layer carries a state `s` and mixes it
with a projection of that state through a scalar gate:

```
out_emb    = W1 @ s                       # state readout
mix_factor = s . (W2 @ out_emb)           # dot product -> scalar
gate       = sigmoid(mix_factor)
next_state = gate * s + (1 - gate) * out_emb
```

The initial state is the embedding of the first token. Logits come from the sum
of every layer's `out_emb`, projected through the output linear:

```
logits = (sum_layers out_emb) @ W_out^T
```

### Frozen, factorised embeddings

`W_in` (`[d, V]`) and `W_out` (`[V, d]`) are initialised from a factorisation of
the bigram count matrix and are **frozen by default** — only the recurrent
layers train.

The rationale: if the residual path is the identity, the network reduces to
`softmax(W_out @ W_in[:, x])`, a bigram predictor. Making that the model's
*default* behaviour means `W_in` and `W_out` should factorise the transition
matrix — an `N x N` matrix decomposed into `N x d` (input embeddings) and
`d x N` (output embeddings). This is also why the two matrices are **never
tied**: they are the two factors, and tying them would force a symmetric
transition matrix.

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

## Development

```bash
uv run pytest
uv run ruff check .
```
