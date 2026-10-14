FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY .devcontainer/post-create.sh .devcontainer/post-create.sh

RUN .devcontainer/post-create.sh

# Keep the CUDA-enabled PyTorch stack from the base image. Install the remaining
# runtime dependencies without asking pip to replace its preinstalled Torch.

COPY . ./

RUN git submodule update --init --recursive

RUN uv sync

RUN python -c 'import torch; arch = torch.cuda.get_arch_list(); print("torch", torch.__version__, "CUDA", torch.version.cuda, "architectures", arch); assert torch.version.cuda is not None and torch.version.cuda >= "12.8"; assert "sm_120" in arch'

ENTRYPOINT ["uv", "run", "myllm.cli"]
CMD ["--help"]