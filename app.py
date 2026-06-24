"""FastAPI service cho Shopee image moderation.

Endpoints:
    GET  /health                      -> trạng thái + device
    POST /moderate      (file upload) -> verdict JSON
    POST /moderate_url  ({"url": ...}) -> verdict JSON

Chạy: uvicorn app:app --host 0.0.0.0 --port 8000
Device: env DEVICE = gpu|cpu|auto (mặc định auto).
"""
import io, os, threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

import moderate as M

app = FastAPI(title="Shopee Moderation Service", version="1.0")
_STATE = {"gpu": False, "ready": False}

# --- Trần concurrency GPU phía server -------------------------------------------------
# Endpoint là sync def -> Starlette chạy trong threadpool (~40 thread). Nếu 1 client bắn
# dồn (vd retry 5xx) thì hàng chục inference GPU chạy song song -> VRAM nhân lên -> OOM.
# Semaphore này CHẶN CỨNG, không phụ thuộc client có tự giới hạn hay không. Mặc định 8
# (= mốc tối ưu trong MODERATION_API.md). Phần dư xếp hàng, không bị từ chối.
_GPU_LIMIT = max(1, int(os.environ.get("GPU_CONCURRENCY", "8")))
_GPU_SEM = threading.Semaphore(_GPU_LIMIT)

# Dọn caching allocator của torch định kỳ -> chống phân mảnh VRAM tích luỹ theo thời gian.
_EMPTY_CACHE_EVERY = max(0, int(os.environ.get("EMPTY_CACHE_EVERY", "200")))
_REQ = {"n": 0}
_REQ_LOCK = threading.Lock()


def _maybe_empty_cache():
    if not _STATE["gpu"] or _EMPTY_CACHE_EVERY == 0:
        return
    with _REQ_LOCK:
        _REQ["n"] += 1
        do = _REQ["n"] % _EMPTY_CACHE_EVERY == 0
    if do:
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


@app.on_event("startup")
def _startup():
    want = os.environ.get("DEVICE", "auto").lower()
    cuda = False
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        pass
    use_gpu = (want == "gpu") or (want == "auto" and cuda)
    if want == "gpu" and not cuda:
        print("[warn] DEVICE=gpu nhưng không thấy CUDA -> chạy CPU")
        use_gpu = False
    M.load_engines(use_gpu=use_gpu)
    _STATE.update(gpu=use_gpu, ready=True)
    print(f"[ready] device={'gpu' if use_gpu else 'cpu'}")


@app.get("/health")
def health():
    return {"status": "ok" if _STATE["ready"] else "loading", "gpu": _STATE["gpu"],
            "gpu_concurrency": _GPU_LIMIT}


def _result(img, nsfw_thr):
    with _GPU_SEM:                                   # ghìm số inference GPU chạy cùng lúc
        r = M.moderate(img, nsfw_thr)
    _maybe_empty_cache()
    # bỏ field detail cồng kềnh (codes/hits) khỏi response gọn; giữ verdict + violations + nsfw
    return {"verdict": r["verdict"], "violations": r["violations"], "nsfw": r["nsfw"]}


@app.post("/moderate")
def moderate_file(file: UploadFile = File(...), nsfw_thr: float = 0.5):
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh")
    return _result(img, nsfw_thr)


class UrlReq(BaseModel):
    url: str
    nsfw_thr: float = 0.5


@app.post("/moderate_url")
def moderate_url(req: UrlReq):
    try:
        return _result(req.url, req.nsfw_thr)   # M.load_image xử lý http URL
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {type(e).__name__}: {e}")
