FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# System packages for pyxis/enroot and rendering
RUN apt-get update && apt-get install -y \
        git tree ffmpeg wget curl \
        libglib2.0-0 libgl1 libglvnd0 libegl1-mesa libgles2-mesa libopengl0 \
        libegl1-mesa-dev libgles2-mesa-dev \
        libosmesa6 libosmesa6-dev \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so

# Create conda environment with Python 3.10
RUN conda create -n openvla-oft python=3.10 -y && conda clean --all --yes

# Put the conda env first on PATH so all pip/python calls use it directly
ENV PATH="/opt/conda/envs/openvla-oft/bin:/opt/conda/bin:${PATH}"
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV NUMBA_DISABLE_JIT=1
ENV PYTHONPATH=/workspace

RUN mkdir -p /workspace
WORKDIR /workspace

# ── Core dependencies (pinned to avoid cross-package breakage) ──────────
RUN pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
            --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir packaging ninja psutil && \
    pip install --no-cache-dir 'flash-attn==2.5.5' --no-build-isolation

# Pin numpy<2 for TensorFlow 2.15 compatibility, pin diffusers + huggingface_hub
# to versions that work with peft==0.11.1
RUN pip install --no-cache-dir \
        'numpy<2' \
        'huggingface_hub==0.23.5' \
        'diffusers==0.27.2' \
        opencv-python-headless==4.11.0.86

# ── OpenVLA-OFT (installs peft==0.11.1, transformers fork, tensorflow, etc.) ─
RUN git clone https://github.com/HuskyKingdom/openvla-oft-yhs && \
    cd openvla-oft-yhs && \
    pip install --no-cache-dir -e .

# ── LIBERO environment ──────────────────────────────────────────────────
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && \
    pip install --no-cache-dir -e LIBERO && \
    cd openvla-oft-yhs && \
    pip install --no-cache-dir -r experiments/robot/libero/libero_requirements.txt

# ── Extra runtime deps (for data conversion and visualization) ──────────
RUN pip install --no-cache-dir \
        av \
        datasets \
        pyarrow \
        scipy

# Re-pin numpy and huggingface_hub in case any of the above steps upgraded them
RUN pip install --no-cache-dir \
        'numpy<2' \
        'huggingface_hub==0.23.5'

CMD ["/bin/bash"]
