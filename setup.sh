#!/usr/bin/env bash
# Chạy nhanh không-Docker trên box Ubuntu GPU (vd Vast.ai). Driver máy tối đa CUDA 12.4.
set -e
sudo apt-get update && sudo apt-get install -y libzbar0 libgl1 libglib2.0-0

# transformers PIN <4.49: bản >=4.50 cần torch>=2.7 (torch.float8_e8m0fnu), mà torch khớp
# driver CUDA 12.4 tối đa là 2.6 (kênh cu124). NSFW ViT không cần transformers mới.
pip install "transformers==4.46.3" accelerate "pillow<12" numpy requests \
    pyzbar opencv-contrib-python rapidocr onnxruntime-gpu ultralytics \
    fastapi "uvicorn[standard]" python-multipart

# QUAN TRỌNG: box hay có sẵn torch CUDA quá mới so với driver, VÀ ultralytics hay kéo torch
# khác đè lên (gây undefined symbol ncclCommResume). ÉP cài lại torch + torchvision cu124 ở
# BƯỚC CUỐI để đồng bộ (torchvision cần cho YOLO person gate).
pip uninstall -y torch torchvision || true
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| GPU available:", torch.cuda.is_available())
PY

echo "Khởi động API tại :8000 ..."
DEVICE=auto uvicorn app:app --host 0.0.0.0 --port 8000
