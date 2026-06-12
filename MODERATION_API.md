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

---

## Response (cả 2 endpoint)
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
- Không có rate-limit phía server → **tự giới hạn** ở client (semaphore/queue).
- Nhiều service cùng bắn → **tự điều phối** để tổng concurrency không vượt 8 (kẻo tranh nhau, latency tăng cho tất cả).

## Tinh chỉnh
- `nsfw_thr` thấp = nhạy hơn (nhiều REJECT), cao = chặt hơn. Mặc định 0.5, dùng thực tế ~0.7.
- Lỗi mạng/timeout → coi như cần xử lý lại (API có thể đang tải / quá tải).
