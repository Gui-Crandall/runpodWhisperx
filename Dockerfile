# Use the updated base CUDA image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set Work Directory
WORKDIR /app

# ARGs and ENVs
ARG WHISPER_MODEL=small
ARG LANG=en
ARG TORCH_HOME=/cache/torch
ARG HF_HOME=/cache/huggingface

# Environment variables
ENV TORCH_HOME=${TORCH_HOME}
ENV HF_HOME=${HF_HOME}
ENV WHISPER_MODEL=${WHISPER_MODEL}
ENV LANG=${LANG}

# Set LD_LIBRARY_PATH for library location (if still necessary)
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=True
ENV DEBIAN_FRONTEND=noninteractive

# Update, upgrade, install packages and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        bash ca-certificates curl file git ffmpeg \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Create and activate virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies, setuptools‑rust, PyTorch, and base tools
RUN pip install --no-cache-dir --upgrade pip==21.* && \
    pip install \
        setuptools-rust==1.8.0 \
        huggingface_hub==0.18.0 \
        runpod==1.3.0 \
        torch==2.0.0 \
        torchvision==0.15.0 \
        torchaudio==2.0.0 \
        -f https://download.pytorch.org/whl/cu118/torch_stable.html

# ────────────────────────────────────────────────────────────────
# WhisperX + diarization stack  (stable “Justin” versions)
# ────────────────────────────────────────────────────────────────
# CHANGED: we now install specific pinned versions instead of the
#          latest GitHub trunk to avoid NumPy‑2 / PyAnnote‑3.3 drift.
RUN pip install --no-cache-dir "whisperx==3.2.0"          # CHANGED
RUN pip install --no-cache-dir "pyannote.audio==3.1.1"    # CHANGED
RUN pip install --no-cache-dir "numpy<2.0"                # CHANGED

# CHANGED: removed old COPY /code + pip install lines.
# They expected a vendored submodule that doesn’t exist in this fork.

# Preload VAD and Whisper models (optional warm‑up)
RUN python -c 'from whisperx.vad import load_vad_model; load_vad_model("cpu");' && \
    python -c 'import faster_whisper; model = faster_whisper.WhisperModel("'${WHISPER_MODEL}'")'

# Preload align model
COPY load_align_model.py .
RUN python load_align_model.py ${LANG}

# Copy and install application-specific requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CHANGED: removed duplicate “pip install pyannote.audio” — already pinned above.

# COPY the example.mp3 file to the container as a default testing audio file
COPY example.mp3 /app/example.mp3
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# Set Stop signal and CMD
STOPSIGNAL SIGINT
CMD ["python", "-u", "handler.py"]
