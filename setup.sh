#!/usr/bin/env bash
# Chạy nhanh không-Docker trên box Ubuntu có sẵn torch+CUDA (vd template PyTorch của Vast.ai).
set -e
sudo apt-get update && sudo apt-get install -y libzbar0 libgl1 libglib2.0-0

pip install "transformers>=4.50.0" accelerate "pillow<12" numpy requests \
    pyzbar opencv-contrib-python rapidocr onnxruntime-gpu \
    fastapi "uvicorn[standard]" python-multipart

# Nếu torch chưa có CUDA, cài bản cu124 (khớp Max CUDA 12.4):
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "Khởi động API tại :8000 ..."
DEVICE=gpu uvicorn app:app --host 0.0.0.0 --port 8000
