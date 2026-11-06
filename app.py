"""FastAPI service cho Shopee image moderation.

Endpoints:
    GET  /health                      -> trạng thái + device
    POST /moderate      (file upload) -> verdict JSON
    POST /moderate_url  ({"url": ...}) -> verdict JSON
    POST /moderate/batch (files[] upload) -> verdict tổng + từng frame (1 round-trip / video)

Chạy: uvicorn app:app --host 0.0.0.0 --port 8000
Device: env DEVICE = gpu|cpu|auto (mặc định auto).
"""
import io, os, threading, time, collections
from typing import List
from fastapi import FastAPI, Request, Response, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

import moderate as M

app = FastAPI(title="Shopee Moderation Service", version="1.0")
_STATE = {"gpu": False, "ready": False}
_BATCH_MAX = max(1, int(os.environ.get("BATCH_MAX_FILES", "16")))


@app.middleware("http")
async def _process_time(request: Request, call_next):
    # Thời gian xử lý PHÍA SERVER (gồm cả xếp hàng chờ semaphore) -> client lấy latency đo được
    # trừ đi số này = phần mạng/upload.
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Process-Ms"] = str(round((time.perf_counter() - t0) * 1000))
    return resp

# --- Trần concurrency GPU phía server -------------------------------------------------
# Endpoint là sync def -> Starlette chạy trong threadpool (~40 thread). Nếu 1 client bắn
# dồn (vd retry 5xx) thì hàng chục inference GPU chạy song song -> VRAM nhân lên -> OOM.
# Trần này CHẶN CỨNG, không phụ thuộc client có tự giới hạn hay không. Mặc định 8
# (= mốc tối ưu trong MODERATION_API.md). Phần dư xếp hàng FIFO; chờ quá QUEUE_TIMEOUT_S -> 503.
_GPU_LIMIT = max(1, int(os.environ.get("GPU_CONCURRENCY", "8")))
_QUEUE_TIMEOUT = float(os.environ.get("QUEUE_TIMEOUT_S", "30"))


class _FairSlots:
    """Semaphore FIFO. threading.Semaphore KHÔNG công bằng: thread mới tới chen được trước thread
    vừa được đánh thức (thread đó quay lại CUỐI hàng) -> dưới tải bão hoà có request bị bỏ đói
    hàng phút tới khi client timeout. Ở đây slot nhả ra được TRAO THẲNG cho người đầu hàng."""

    def __init__(self, n):
        self._n, self.busy = n, 0
        self._q = collections.deque()
        self._lock = threading.Lock()

    def acquire(self, timeout):
        with self._lock:
            if self.busy < self._n and not self._q:
                self.busy += 1
                return True
            ev = threading.Event()
            self._q.append(ev)
        if ev.wait(timeout):
            return True
        with self._lock:
            if ev.is_set():                              # được trao slot đúng lúc vừa hết giờ
                return True
            self._q.remove(ev)
            return False

    def release(self):
        with self._lock:
            if self._q:
                self._q.popleft().set()                  # trao slot, busy giữ nguyên
            else:
                self.busy -= 1

    @property
    def waiting(self):
        return len(self._q)


_GPU_SLOTS = _FairSlots(_GPU_LIMIT)

# Dọn caching allocator của torch định kỳ -> chống phân mảnh VRAM tích luỹ theo thời gian.
_EMPTY_CACHE_EVERY = max(0, int(os.environ.get("EMPTY_CACHE_EVERY", "200")))
_REQ = {"n": 0}
_REQ_LOCK = threading.Lock()


def _maybe_empty_cache(frames=1):
    if not _STATE["gpu"] or _EMPTY_CACHE_EVERY == 0:
        return
    with _REQ_LOCK:
        before = _REQ["n"]
        _REQ["n"] += frames                              # đếm theo frame -> batch không làm thưa nhịp dọn
        do = _REQ["n"] // _EMPTY_CACHE_EVERY != before // _EMPTY_CACHE_EVERY
    if do:
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


@app.on_event("startup")
async def _threadpool_size():
    # Endpoint sync chạy trong threadpool anyio (mặc định 40 thread/worker). Request đợi slot GPU
    # vẫn giữ 1 thread -> >40 kết nối dồn vào 1 worker thì phần dư kẹt ở đây, NGOÀI hàng đợi FIFO
    # (không timeout, không 503). Nới rộng để mọi request vào được hàng đợi có kiểm soát.
    import anyio.to_thread
    anyio.to_thread.current_default_thread_limiter().total_tokens = int(os.environ.get("THREADPOOL_SIZE", "128"))


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
            "gpu_concurrency": _GPU_LIMIT, "batch_max_files": _BATCH_MAX,
            "gpu_busy": _GPU_SLOTS.busy, "queue_waiting": _GPU_SLOTS.waiting,
            "queue_timeout_s": _QUEUE_TIMEOUT, "cpu_threads": M.cpu_threads,
            "ocr_providers": M.ocr_providers, "vram_budget": M.vram_budget}


# Lỗi hết VRAM (torch OOM, ORT "Failed to allocate", cuDNN alloc/internal error khi GPU đầy) là
# lỗi TẠM -> trả 503 + Retry-After để client retry, thay vì 500 như lỗi logic.
_OOM_MARKERS = ("out of memory", "outofmemory", "failed to allocate", "cudnn_status_internal_error",
                "cudnn_status_alloc_failed", "cublas_status_alloc_failed")


def _run_gpu(response, fn, *args):
    t0 = time.perf_counter()
    if not _GPU_SLOTS.acquire(_QUEUE_TIMEOUT):       # ghìm số inference GPU chạy cùng lúc
        raise HTTPException(status_code=503, headers={"Retry-After": "2"},
                            detail=f"Hàng đợi GPU quá {_QUEUE_TIMEOUT:.0f}s "
                                   f"({_GPU_SLOTS.waiting} request đang chờ), thử lại sau")
    # Thời gian chờ slot: X-Process-Ms − X-Queue-Ms ≈ thời gian xử lý thật.
    response.headers["X-Queue-Ms"] = str(round((time.perf_counter() - t0) * 1000))
    try:
        return fn(*args)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if not any(m in msg.lower() for m in _OOM_MARKERS):
            raise
        try:
            import torch; torch.cuda.empty_cache()       # trả cache torch lại cho ORT/worker khác
        except Exception:
            pass
        print(f"[oom] {msg[:300]}")
        raise HTTPException(status_code=503, detail=f"GPU hết VRAM, thử lại: {msg[:200]}",
                            headers={"Retry-After": "1"})
    finally:
        _GPU_SLOTS.release()


def _result(response, img, nsfw_thr):
    r = _run_gpu(response, M.moderate, img, nsfw_thr)
    _maybe_empty_cache()
    # bỏ field detail cồng kềnh (codes/hits) khỏi response gọn; giữ verdict + violations + nsfw
    return {"verdict": r["verdict"], "violations": r["violations"], "nsfw": r["nsfw"]}


@app.post("/moderate")
def moderate_file(response: Response, file: UploadFile = File(...), nsfw_thr: float = 0.5):
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh")
    return _result(response, img, nsfw_thr)


class UrlReq(BaseModel):
    url: str
    nsfw_thr: float = 0.5


@app.post("/moderate_url")
def moderate_url(req: UrlReq, response: Response):
    try:
        return _result(response, req.url, req.nsfw_thr)   # M.load_image xử lý http URL
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {type(e).__name__}: {e}")


@app.post("/moderate/batch")
def moderate_batch(response: Response, files: List[UploadFile] = File(...), nsfw_thr: float = 0.5,
                   stop_on_reject: bool = True):
    """Nhiều frame của CÙNG 1 video trong 1 request. Giữ 1 slot semaphore cho cả batch
    (bên trong OCR chạy tuần tự -> đỉnh VRAM mỗi slot ~ 1 OCR + 1 NSFW batch)."""
    if len(files) > _BATCH_MAX:
        raise HTTPException(status_code=413, detail=f"Tối đa {_BATCH_MAX} file / request")
    imgs = []
    for i, f in enumerate(files):
        try:
            imgs.append(Image.open(io.BytesIO(f.file.read())).convert("RGB"))
        except Exception:
            raise HTTPException(status_code=400, detail=f"Không đọc được ảnh index={i}")
    rs = _run_gpu(response, M.moderate_batch, imgs, nsfw_thr, stop_on_reject)
    _maybe_empty_cache(len(imgs))
    results = [{"index": i, "verdict": r["verdict"], "violations": r["violations"], "nsfw": r["nsfw"]}
               for i, r in enumerate(rs)]
    rejects = [r for r in results if r["verdict"] == "REJECT"]
    violations = sorted({v for r in rejects for v in r["violations"].split(",") if v})
    return {"verdict": "REJECT" if rejects else "ACCEPT",
            "violations": ",".join(violations),
            "reject_index": rejects[0]["index"] if rejects else None,
            "results": results}
