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

## Development

```bash
uv run pytest
uv run ruff check .
```
