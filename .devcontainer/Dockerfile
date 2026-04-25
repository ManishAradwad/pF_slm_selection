# Downgraded to 12.4 to allow pre-built wheels
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages. Python 3.11 via deadsnakes.
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

# python3 / python / pip resolve to 3.11.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip

# PyTorch CUDA 12.4 (Changed from cu126 to cu124)
RUN pip install --no-cache-dir torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# Install from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install the pre-built wheel directly (No compiling, no C++ hacks needed!)
RUN pip install --no-cache-dir llama-cpp-python==0.3.20 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
