# Moderation API — thay cho OpenAI Moderation

Dịch vụ kiểm duyệt **ảnh / frame video** (self-host GPU). Gửi 1 ảnh → trả verdict.
Phát hiện: **NSFW**, **QR/Barcode**, **TikTok**, **nền tảng đối thủ** (Facebook/Lazada/...).

- Base URL: `http://91.150.160.38:16815`  *(đổi theo deploy thực tế)*
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

## Tinh chỉnh
- `nsfw_thr` thấp = nhạy hơn (nhiều REJECT), cao = chặt hơn. Mặc định 0.5, dùng thực tế ~0.7.
- Lỗi mạng/timeout → coi như cần xử lý lại (API có thể đang tải / quá tải).
