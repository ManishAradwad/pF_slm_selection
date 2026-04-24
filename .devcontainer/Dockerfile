# Use the NVIDIA CUDA compiler image (Ubuntu 22.04 base)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts from stalling the build
ENV DEBIAN_FRONTEND=noninteractive

# Install core tools, add the Python 3.11 repository, and install Python
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default system Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace

# Force llama-cpp-python to compile with CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

# Upgrade pip and install the lm-evaluation-harness with GGUF support
RUN python3 -m pip install --upgrade pip
RUN pip3 install git+https://github.com/EleutherAI/lm-evaluation-harness.git
RUN pip3 install "lm_eval[gguf]"