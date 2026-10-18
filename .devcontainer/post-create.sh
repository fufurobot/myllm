#!/usr/bin/env bash
# Codespaces / devcontainer provisioning for myllm.
#
# Runs after the container is created:
#   1. Installs uv.
#   2. Installs the Python dependencies (CPU or CUDA 11.8 torch backend).
#   3. Downloads models/datasets declared by .gitkeep markers via download-hf.sh.
#
# Configure with environment variables:
#   TORCH_BACKEND     "cpu" or "cu118"                 (default: cpu)
#   INSTALL_DEV       "true" to include dev deps       (default: true)
#   DOWNLOAD_DATASETS "false" to skip HF downloads     (default: true)
#   HFD_EXTRA_ARGS    Extra args forwarded to hfd.sh   (e.g. "--hf_token xxx")

set -euo pipefail

WORKSPACE="${containerWorkspaceFolder:-$(pwd)}"
cd "$WORKSPACE"

TORCH_BACKEND="${TORCH_BACKEND:-cpu}"
INSTALL_DEV="${INSTALL_DEV:-true}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-true}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning: %s\033[0m\n' "$*" >&2; }

case "$TORCH_BACKEND" in
    cpu|cu118) ;;
    *) echo "Error: TORCH_BACKEND must be 'cpu' or 'cu118', got '$TORCH_BACKEND'" >&2; exit 1 ;;
esac

# ---- Install aria2 (static binary) ----
log "Installing aria2"
mkdir -p ~/.local/bin
curl -L https://github.com/P3TERX/aria2-builder/releases/download/1.35.0_2020.09.04/aria2-1.35.0-static-linux-amd64.tar.gz \
  | tar xz -C ~/.local/bin --strip-components=1

# ---- Install uv ----
log "Installing uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# ---- Make sure ~/.local/bin and uv's bin dir are on PATH ----
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
# uv installs to ~/.local/bin by default; also source its env file if present
[ -f "$HOME/.local/env" ] && . "$HOME/.local/env"

log "Done. Versions:"
aria2c --version | head -n1 || true
uv --version || true

# The uv installer drops the binary in ~/.local/bin, which may not be on PATH
# for non-login shells yet.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv installation failed or uv is not on PATH" >&2
    exit 1
fi
uv --version

# --- 2. Install dependencies -------------------------------------------------
# Codespaces GPU-less runners must use the CPU wheels. Set TORCH_BACKEND=cu118
# to install the CUDA 11.8 build instead.
log "Installing Python dependencies (torch backend: $TORCH_BACKEND)"
if [[ "$INSTALL_DEV" == "true" ]]; then
    uv sync --extra "$TORCH_BACKEND"
else
    uv sync --extra "$TORCH_BACKEND" --no-dev
fi

log "Verifying torch installation"
uv run --no-sync python - <<'PY'
import torch
print(f"torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.version.cuda:
    print(f"built for CUDA {torch.version.cuda}")
PY

# --- 3. Download models and datasets ----------------------------------------
if [[ "$DOWNLOAD_DATASETS" != "true" ]]; then
    log "Skipping dataset download (DOWNLOAD_DATASETS=$DOWNLOAD_DATASETS)"
    exit 0
fi

log "Downloading models and datasets"

if [[ ! -f .env && -f .env.example ]]; then
    cp .env.example .env
    warn "Created .env from .env.example — add your HF_TOKEN there for gated repos."
fi

# download-hf.sh expects hfd.sh to be executable and present at the repo root.
if [[ ! -x ./hfd.sh ]]; then
    chmod +x ./hfd.sh 2>/dev/null || true
fi

if [[ ! -x ./hfd.sh ]]; then
    warn "hfd.sh missing or not executable; skipping dataset download."
    exit 0
fi

chmod +x ./download-hf.sh

# The download shell is best-effort: a missing/gated dataset must not break the
# whole Codespace creation, so failures are reported but tolerated.
if ./download-hf.sh; then
    log "Dataset download complete."
else
    warn "download-hf.sh failed. Re-run it manually after fixing the cause:"
    warn "  HF_TOKEN=... ./download-hf.sh"
fi
