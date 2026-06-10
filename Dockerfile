# Base có sẵn torch + CUDA 12.4 + cuDNN 9 (khớp Max CUDA 12.4 của máy Vast.ai RTX 3060).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVICE=gpu

# Thư viện hệ thống: zbar (pyzbar) + GL/glib (opencv).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libzbar0 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (torch/cuda/cudnn đã có trong base: torch 2.5.1 / CUDA 12.4).
# transformers PIN <4.49: bản mới cần torch>=2.7 (float8_e8m0fnu) -> vỡ với torch 2.5/2.6.
RUN pip install --no-cache-dir \
        "transformers==4.46.3" accelerate "pillow<12" numpy requests \
        pyzbar opencv-contrib-python rapidocr onnxruntime-gpu \
        fastapi "uvicorn[standard]" python-multipart

COPY moderate.py app.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
