# Moderation API — thay cho OpenAI Moderation

Dịch vụ kiểm duyệt **ảnh / frame video** (self-host GPU). Gửi 1 ảnh → trả verdict.
Phát hiện: **NSFW**, **QR/Barcode**, **TikTok**, **nền tảng đối thủ** (Facebook/Lazada/...).

- Base URL: Lấy từ infisical, key AI_MODERATION_URL trong shared (có thể xem cách lấy ở worker_upload) (example: http://91.150.160.38:16815)
- Không cần API key.

---

## Endpoints

### `GET /health`
```json
{ "status": "ok", "gpu": true }
```

### `POST /moderate` — upload file ảnh
Form-data: `file` = ảnh. Query (tuỳ chọn): `nsfw_thr` (mặc định 0.5; khuyến nghị 0.7).

```bash
curl -F "file=@frame.jpg" "http://91.150.160.38:16815/moderate?nsfw_thr=0.7"
```

### `POST /moderate_url` — gửi URL ảnh
```bash
curl -X POST http://91.150.160.38:16815/moderate_url \
     -H "Content-Type: application/json" \
     -d '{"url": "https://.../anh.jpg", "nsfw_thr": 0.7}'
```

### `POST /moderate/batch` — nhiều frame của 1 video trong 1 request
Form-data: `files` lặp lại N lần (tối đa `BATCH_MAX_FILES`, mặc định 16). Query: `nsfw_thr`,
`stop_on_reject` (mặc định `true`). Server chạy theo stage: QR mọi frame (CPU, song song) →
NSFW gộp 1 batch GPU → OCR từng frame. `stop_on_reject=true`: có frame REJECT thì các frame còn lại
bỏ stage đắt hơn và trả `SKIPPED` (giống client dừng sớm khi gửi tuần tự).

```bash
curl -F "files=@f0.jpg" -F "files=@f1.jpg" -F "files=@f2.jpg"      "http://<host>/moderate/batch?nsfw_thr=0.7"
```
```json
{ "verdict": "REJECT", "violations": "tiktok", "reject_index": 1,
  "results": [
    {"index": 0, "verdict": "ACCEPT",  "violations": "",       "nsfw": 0.01},
    {"index": 1, "verdict": "REJECT",  "violations": "tiktok", "nsfw": 0.02},
    {"index": 2, "verdict": "SKIPPED", "violations": "",       "nsfw": 0.03}
  ] }
```
- `verdict` tổng: `REJECT` nếu có ≥1 frame REJECT, ngược lại `ACCEPT`.
- `violations` tổng: hợp các loại vi phạm (sort, ngăn bởi dấu phẩy). `reject_index`: frame REJECT đầu tiên (theo index) hoặc `null`.
- `results[i].verdict`: `ACCEPT` / `REJECT` / `SKIPPED` (không chấm hết vì đã có frame khác REJECT).
- `nsfw` của frame bị loại ở QR / skip trước NSFW là `null`.
- Lỗi: 400 nếu 1 file không đọc được ảnh (`detail` có index), 413 nếu quá số file.

Mọi response có header **`X-Process-Ms`** = thời gian phía server (gồm chờ hàng đợi GPU).
Latency client đo được − `X-Process-Ms` ≈ mạng + upload.

---

## Response (`/moderate`, `/moderate_url`)
```json
{ "verdict": "REJECT", "violations": "nsfw", "nsfw": 0.94 }
```
| Field | Ý nghĩa |
|-------|---------|
| `verdict` | `"REJECT"` (vi phạm) hoặc `"ACCEPT"` (sạch) |
| `violations` | chuỗi loại vi phạm, ngăn bởi dấu phẩy. Rỗng nếu ACCEPT. Giá trị: `nsfw`, `qr_barcode`, `tiktok`, `competitor` (có thể ghép, vd `"tiktok,nsfw"`) |
| `nsfw` | điểm NSFW 0–1 (hoặc `null` nếu bị loại ở bước QR trước khi chấm NSFW) |

→ Quy tắc đơn giản: **`verdict == "REJECT"` nghĩa là CHẶN.**

---

## Ví dụ Python
```python
import requests

API = "http://91.150.160.38:16815"

def is_blocked(image_path, nsfw_thr=0.7):
    with open(image_path, "rb") as f:
        r = requests.post(f"{API}/moderate",
                          files={"file": ("img.jpg", f, "image/jpeg")},
                          params={"nsfw_thr": nsfw_thr}, timeout=60)
    d = r.json()
    return d["verdict"] == "REJECT", d["violations"], d["nsfw"]

blocked, reason, score = is_blocked("anh.jpg")
if blocked:
    print("CHẶN:", reason, score)
```

---

## Video
API chỉ nhận **ảnh**. Với video: **client tự trích frame** rồi gọi `/moderate` từng frame.
Gặp frame REJECT đầu tiên → coi như video vi phạm. (Xem `moderate_video_client.py`,
`check_one_video.py` trong repo.)

```python
import cv2, requests
cap = cv2.VideoCapture(VIDEO_URL)         # đọc thẳng URL được
total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
for frac in (0.1, 0.25, 0.5, 0.75, 0.9):  # lấy 5 frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac*total)); ok, fr = cap.read()
    if not ok: continue
    _, buf = cv2.imencode(".jpg", fr)
    d = requests.post(f"{API}/moderate?nsfw_thr=0.7",
                      files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")}).json()
    if d["verdict"] == "REJECT":
        print("Video vi phạm:", d["violations"], "@", f"{frac*100:.0f}%"); break
```

---

## Chuyển từ OpenAI Moderation
| OpenAI | API này |
|--------|---------|
| `POST /v1/moderations` (text) | `POST /moderate` (ảnh) |
| `results[0].flagged` (bool) | `verdict == "REJECT"` |
| `results[0].categories` | `violations` (chuỗi) |
| `category_scores.sexual` | `nsfw` (điểm) |

Khác biệt chính: API này nhận **ảnh** (không phải text), tự host nên **không tính tiền theo request**,
và có thêm loại QR/TikTok/đối thủ đặc thù cho TMĐT.

## Tải & giới hạn (BẮT BUỘC đọc nếu tích hợp)

Server là **1 instance, 1 GPU (RTX 3060)** — **DÙNG CHUNG** cho mọi service.

Số đo thực tế (64 ảnh, workload trộn):

| Concurrency | Throughput | s/ảnh |
|---|---|---|
| 1 | ~0.5 ảnh/s | 1.96 |
| **8 (tối ưu)** | **~4 ảnh/s** (~240/phút) | 0.24 |
| 16 | ~3.6 ảnh/s (chậm hơn) | 0.275 |

**Quy tắc cho service tích hợp:**
- **Giới hạn ≤ 8 request đồng thời** (TỔNG của tất cả service cộng lại, không phải mỗi service). Vượt 8 **không nhanh hơn**, chỉ tăng độ trễ.
- **Trần throughput ≈ 4 ảnh/giây** (~14k ảnh/giờ). Cần hơn → phải scale thêm instance.
- 1 request lẻ: ~0.25–2s (gồm mạng). Đặt **timeout 60s**, **retry** khi lỗi mạng / 5xx.
- Server **tự chặn cứng** concurrency GPU bằng semaphore (mặc định 8) → request dư **xếp hàng**, không OOM. Client vẫn nên tự giới hạn để latency thấp.
- Nhiều service cùng bắn → **tự điều phối** để tổng concurrency không vượt 8 (kẻo tranh nhau, latency tăng cho tất cả).

## Tinh chỉnh
- `nsfw_thr` thấp = nhạy hơn (nhiều REJECT), cao = chặt hơn. Mặc định 0.5, dùng thực tế ~0.7.
- Lỗi mạng/timeout → coi như cần xử lý lại (API có thể đang tải / quá tải).

### Env tinh chỉnh phía server (chống VRAM phình / OOM)
- `GPU_CONCURRENCY` (mặc định 8): trần inference GPU chạy song song. Hạ xuống nếu GPU nhỏ.
- `MAX_OCR_SIDE` (mặc định 1280): cap max-side ảnh đẩy vào OCR, cắt đỉnh VRAM. 0 = tắt cap.
- `EMPTY_CACHE_EVERY` (mặc định 200): mỗi N request gọi `torch.cuda.empty_cache()` chống phân mảnh. 0 = tắt.
- `OCR_GPU_MEM_LIMIT` (bytes, optional): trần VRAM cho OCR onnxruntime, vd `2147483648` = 2GB.
- `UVICORN_WORKERS` (mặc định 1): số process uvicorn. Mỗi process load model riêng và có semaphore `GPU_CONCURRENCY` riêng → tăng khi CPU còn rảnh mà GIL nghẽn. Khuyến nghị: GPU 16GB → **2** worker (3 worker từng OOM ở ≥36 request batch đồng thời); 24GB → 3.
- VRAM tự chia theo worker (`VRAM_BUDGET=1` mặc định): mỗi worker được `VRAM_USABLE`(0.92) × VRAM / `UVICORN_WORKERS`; torch (NSFW) trần `TORCH_VRAM_GB` (1.5), OCR onnxruntime nhận phần còn lại trừ `CUDA_CTX_RESERVE_GB` (0.7) làm `gpu_mem_limit` (trừ khi đặt `OCR_GPU_MEM_LIMIT`). Xem `GET /health` → `vram_budget`, `ocr_providers` (phải chứa `CUDAExecutionProvider`).
- Hết VRAM giữa chừng → **503 + `Retry-After: 1`** (lỗi tạm, client nên retry), không còn 500.
- Hàng đợi GPU là **FIFO**; request chờ slot quá `QUEUE_TIMEOUT_S` (mặc định 30) → **503 + `Retry-After: 2`** thay vì treo tới khi client timeout. Header **`X-Queue-Ms`** = thời gian chờ slot (X-Process-Ms − X-Queue-Ms ≈ xử lý thật). `/health` có `gpu_busy`, `queue_waiting`.
- `THREADPOOL_SIZE` (mặc định 128): thread anyio mỗi worker (mặc định của Starlette là 40; request vượt mức đó kẹt ngoài hàng đợi, không có timeout).
- CPU tự chia theo worker từ quota cgroup thật (không phải số lõi host): torch + onnxruntime intra-op + pool QR mỗi worker dùng `quota / UVICORN_WORKERS` thread; cv2 1 thread. Ghi đè bằng `CPU_THREADS`. Xem `/health` → `cpu_threads`.
- `UVICORN_KEEPALIVE` (mặc định 75s): keep-alive timeout của uvicorn (mặc định gốc 5s gây "Connection aborted" khi client dùng lại kết nối rảnh).
- `BATCH_MAX_FILES` (mặc định 16): số frame tối đa / request `/moderate/batch`.
- `QR_THREADS` (mặc định min(8, số CPU)): pool quét QR song song trong 1 batch.
