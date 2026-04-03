FROM nvcr.io/nvidia/jax:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# System deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# TensorFlow (CPU only, for TFDS)
RUN pip install --upgrade pip && pip install \
    tensorflow-cpu==2.14.0 \
    tensorflow-probability==0.22.0 \
    tensorflow-datasets==4.9.4

# Project dependencies
RUN pip install \
    scipy==1.12.0 \
    matplotlib \
    scikit-learn \
    tqdm \
    einops \
    jaxtyping \
    absl-py \
    ml-collections \
    opt-einsum \
    "diffusers>=0.30.0" \
    transformers \
    wandb \
    tabulate \
    Pillow \
    datasets \
    typeguard

# chex only — flax 0.8.2 and optax 0.2.2 are already in the NGC base image
RUN pip install "chex"

WORKDIR /workspace
