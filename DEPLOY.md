# Deploy lên Vast.ai (RTX 3060, CUDA 12.4)

API: `POST /moderate` (upload ảnh) hoặc `POST /moderate_url` ({"url": ...}) → `{verdict, violations, nsfw}`.

## Cách A — Docker image build trên GitHub Actions (không cần Docker ở máy)

Workflow `.github/workflows/build-docker.yml` build + push image lên GHCR.

**1. Chạy build:**
- Tự chạy khi push thay đổi `Dockerfile`/`app.py`/`moderate.py`, HOẶC vào tab **Actions** → **build-docker** → **Run workflow**.
- Xong: image ở `ghcr.io/spy-solution/moderation-service:latest`.

**2. Cho phép Vast.ai pull (1 lần):** GitHub → repo → **Packages** → `moderation-service` → **Package settings** → **Change visibility → Public**. (Hoặc để private rồi đưa credentials cho Vast.)

**3. Tạo instance trên Vast.ai:**
- Image: `ghcr.io/spy-solution/moderation-service:latest`
- Mở port **8000** (Docker options: `-p 8000:8000`).
- Launch. Container tự chạy `uvicorn ... :8000`.

> Nếu có Docker ở máy thì build tay cũng được: `docker build -t <user>/moderation:gpu . && docker push ...`

**3. Lần khởi động đầu** sẽ tự tải model (~400MB: NSFW + RapidOCR + WeChat). Sau đó sẵn sàng.

## Cách B — Không Docker (nhanh để thử)

SSH vào instance (chọn template PyTorch CUDA 12.4 của Vast.ai), rồi:
```bash
git clone https://github.com/Spy-Solution/moderation-service.git
cd moderation-service
bash setup.sh
```
`setup.sh` cài deps hệ thống + python, rồi chạy API ở `:8000`.

## Kiểm tra

```bash
# health
curl http://<IP>:<PORT>/health
# -> {"status":"ok","gpu":true}

# duyệt 1 ảnh
curl -F "file=@sample/tiktok-1.jpg" http://<IP>:<PORT>/moderate
# -> {"verdict":"REJECT","violations":"tiktok","nsfw":0.0}

# duyệt theo URL
curl -X POST http://<IP>:<PORT>/moderate_url \
     -H "Content-Type: application/json" \
     -d '{"url":"https://.../anh.jpg"}'
```

> IP/PORT: Vast.ai map port 8000 ra một cổng public — xem ở tab **Instances** (dạng `IP:port`).

## Lưu ý

- **Disk 32GB**: image PyTorch base ~7GB + deps ~2GB + model ~0.4GB → vừa đủ.
- **CUDA**: máy Max CUDA 12.4 → base `cuda12.4-cudnn9` + `onnxruntime-gpu`. Nếu onnxruntime báo lỗi cuDNN, pin version: `pip install onnxruntime-gpu==1.19.2`.
- **QR/WeChat** chạy CPU (opencv pip không có CUDA) — không ảnh hưởng, vốn nhanh.
- Verify GPU sau khi chạy: `curl /health` phải trả `"gpu": true`. Nếu `false`:
  - **Hay gặp ở Cách B**: box có sẵn torch build CUDA quá mới so với driver (vd torch CUDA13 vs driver 12.4) → log báo `NVIDIA driver too old (found version 12040)`. Sửa: `pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu124`, rồi chạy lại. (Cách A Docker không dính vì base image torch khớp CUDA 12.4.)
- Service hiện xử lý **ảnh**. Video (trích frame) là bước sau.
