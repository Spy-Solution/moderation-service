# Base có sẵn torch + CUDA 12.4 + cuDNN 9 (khớp Max CUDA 12.4 của máy Vast.ai RTX 3060).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    DEVICE=gpu \
    GPU_CONCURRENCY=12 \
    MOD_TIMING=0
# GPU_CONCURRENCY=12: hợp Tesla P100 16GB. Trên GPU nhỏ hơn (vd 3060 12GB) hạ về 8.
# MOD_TIMING=0: bật =1 (qua -e) khi cần log thời gian từng stage (qr/nsfw/ocr) để chẩn đoán.
# QR mặc định đã nhẹ (xem moderate.py). Bật mạnh hơn: -e QR_FULL_VARIANTS=1 / QR_WECHAT=1.
# Override per-instance khỏi rebuild: thêm "-e GPU_CONCURRENCY=8" vào Docker options trên Vast.

# Thư viện hệ thống: zbar (pyzbar) + GL/glib (opencv).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libzbar0 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (torch/cuda/cudnn đã có trong base: torch 2.5.1 / CUDA 12.4).
# transformers PIN <4.49: bản mới cần torch>=2.7 (float8_e8m0fnu) -> vỡ với torch 2.5/2.6.
# rapidocr kéo opencv-python về -> gỡ rồi cài opencv-contrib-python (superset, có WeChat QR)
# để tránh 2 bản cv2 đụng nhau.
# onnxruntime-gpu PIN: bản mới nhất build cho CUDA 13 (libcudart.so.13) -> vỡ với base
# CUDA 12.4 (chỉ có libcudart.so.12). 1.20.1 build cho CUDA 12.x + cuDNN 9, khớp base.
RUN pip install --no-cache-dir \
        "transformers==4.46.3" accelerate "pillow<12" numpy requests \
        pyzbar rapidocr "onnxruntime-gpu==1.20.1" \
        fastapi "uvicorn[standard]" python-multipart \
    && pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true \
    && pip install --no-cache-dir opencv-contrib-python

COPY moderate.py app.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8000
# entrypoint.sh thêm torch/lib + nvidia/*/lib vào LD_LIBRARY_PATH để onnxruntime thấy CUDA/cuDNN.
CMD ["./entrypoint.sh"]
