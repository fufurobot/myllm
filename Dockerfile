FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Keep the CUDA-enabled PyTorch stack from the base image. Install the remaining
# runtime dependencies without asking pip to replace its preinstalled Torch.
RUN python -m pip install --no-cache-dir \
    "accelerate>=1.15.0" \
    "huggingface-hub>=0.36.2" \
    "rwkv7-hf>=0.9.0" \
    "transformers>=4.57.6"

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -c 'import torch; arch = torch.cuda.get_arch_list(); print("torch", torch.__version__, "CUDA", torch.version.cuda, "architectures", arch); assert torch.version.cuda is not None and torch.version.cuda >= "12.8"; assert "sm_120" in arch'

ENTRYPOINT ["python", "-m", "myllm.cli"]
CMD ["--help"]