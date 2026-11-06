#!/usr/bin/env bash
# onnxruntime-gpu cần libcudnn/libcublas/libcudart mà CUDA EP nạp bằng dlopen lúc tạo session.
# Trong image PyTorch, các lib này nằm trong thư mục gói torch (torch/lib) + /opt/conda/lib,
# KHÔNG trên đường dẫn mặc định -> onnxruntime không thấy -> rớt CPU âm thầm.
# Phải set LD_LIBRARY_PATH TRƯỚC khi Python chạy (dlopen đọc env lúc process khởi động).
set -e
TORCH_LIB="$(python -c 'import os,torch; print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
# nvidia-* pip packages (cudnn/cublas...) nếu có, mỗi gói 1 thư mục lib riêng.
NV_LIBS="$(python -c 'import os,glob,site
ds=[]
for sp in site.getsitepackages():
    ds+=glob.glob(os.path.join(sp,"nvidia","*","lib"))
print(":".join(ds))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${NV_LIBS}:/opt/conda/lib:${LD_LIBRARY_PATH}"
echo "[entrypoint] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
# UVICORN_WORKERS>1: nhiều process (mỗi process tự load model, ~1-1.5GB VRAM) -> vượt GIL cho
# phần CPU (QR, decode, preprocess). GPU_CONCURRENCY là trần RIÊNG của từng worker.
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers "${UVICORN_WORKERS:-1}"
