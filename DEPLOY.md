# Deploy lên Vast.ai (RTX 3060, CUDA 12.4)

API: `POST /moderate` (upload ảnh) hoặc `POST /moderate_url` ({"url": ...}) → `{verdict, violations, nsfw}`.

## Cách A — Docker (khuyến nghị, tái lập được)

**1. Build + push image** (máy local có Docker):
```bash
docker build -t <dockerhub_user>/moderation:gpu .
docker push <dockerhub_user>/moderation:gpu
```

**2. Tạo instance trên Vast.ai:**
- Image: `<dockerhub_user>/moderation:gpu`
- Mở port **8000** (Docker options: `-p 8000:8000`).
- Launch. Container tự chạy `uvicorn ... :8000`.

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
- Verify GPU sau khi chạy: `curl /health` phải trả `"gpu": true`. Nếu `false` → torch không thấy CUDA trong container (sai base image / driver).
- Service hiện xử lý **ảnh**. Video (trích frame) là bước sau.
