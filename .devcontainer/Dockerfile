# CUDA 12.6.3 dev image — provides nvcc and headers for building llama.cpp
# with CUDA. CUDA 12.6 is PyTorch's current stable target (cu126); 12.8 also
# supported but newer-stack. The host driver (591.74 / max CUDA 13.1) is
# well within range. Ubuntu 22.04 (glibc 2.35) is required for prebuilt CUDA
# wheels — the prior python:3.11-bullseye base shipped glibc 2.31, which is
# why llama-cpp-python had to fall back to CPU-only.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages. Python 3.11 via deadsnakes (Ubuntu 22.04 ships 3.10).
# cmake + ninja-build are needed for the from-source llama-cpp-python build:
# scikit-build-core invokes cmake (to configure llama.cpp's CUDA build) and
# ninja (to actually compile) — build-essential covers gcc/make but not these.
RUN apt-get update && apt-get install -y \
        software-properties-common \
        curl \
        git \
        build-essential \
        cmake \
        ninja-build \
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

# torch from PyTorch's CUDA 12.6 wheel index. Per pytorch.org/get-started,
# the official pattern is --index-url (NOT --extra-index-url) which restricts
# pip to *only* the PyTorch index for these packages. Other deps come from
# PyPI in the next step. Separate layer so requirements.txt tweaks don't
# invalidate this large download.
RUN pip install --no-cache-dir torch==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu126

# Remaining pip deps from PyPI.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# llama-cpp-python — from-source build with CUDA.
# Why --no-binary: abetlen ships prebuilt CUDA wheels for cu121-cu125 only,
# nothing for our cu126 base. Without --no-binary, pip would silently fall
# back to a CPU-only wheel from PyPI and every eval would run on CPU. The
# CMAKE_ARGS env tells llama.cpp's CMakeLists to enable the CUDA backend.
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir --no-binary=llama-cpp-python llama-cpp-python==0.3.20
