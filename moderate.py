#!/usr/bin/env python
"""Shopee image moderation (CPU): QR/Barcode -> NSFW -> OCR keyword.

Chạy:
    python moderate.py <ảnh hoặc folder> [--workers N] [--csv out.csv] [--nsfw-thr 0.5]

Pipeline trong moderate(): QR/barcode -> NSFW -> OCR, theo thứ tự chi phí, DỪNG SỚM.
Engine: pyzbar+cv2+WeChat (QR), LukeJacob2023/nsfw-image-detector (NSFW), RapidOCR (OCR).
"""
from __future__ import annotations
import os, sys, io, re, glob, time, csv, argparse, unicodedata, pathlib, urllib.request

PROJ = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJ / ".hf_cache"))          # cache model -> project (off C:)

import numpy as np
from PIL import Image

# ======================================================================================
# ENGINES (load 1 lần)
# ======================================================================================
nsfw_clf = None        # transformers pipeline
ocr_engine = None      # RapidOCR
_OCR_NEW = True
wechat_detector = None # cv2 WeChat QR (optional)

import cv2  # opencv-contrib-python (cv2 base + wechat_qrcode)


def _make_ocr_gpu(RapidOCR):
    """Thử bật CUDA EP cho RapidOCR qua vài kiểu param (khác nhau theo version)."""
    for params in ({"EngineConfig.onnxruntime.use_cuda": True},
                   {"Global.use_cuda": True}):
        try:
            return RapidOCR(params=params)
        except Exception:
            pass
    try:
        return RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)  # API cũ
    except Exception:
        return None


def load_engines(use_gpu: bool = False, quiet: bool = False):
    global nsfw_clf, ocr_engine, _OCR_NEW, wechat_detector

    def log(*a):
        if not quiet:
            print(*a, file=sys.stderr)

    # --- NSFW (5 lớp: drawings/hentai/neutral/porn/sexy) ---
    from transformers import (pipeline, AutoModelForImageClassification,
                              AutoImageProcessor, ViTImageProcessor)
    mid = "LukeJacob2023/nsfw-image-detector"
    mdl = AutoModelForImageClassification.from_pretrained(mid)
    try:
        ip = AutoImageProcessor.from_pretrained(mid)
    except Exception:
        ip = ViTImageProcessor.from_pretrained(mid)          # repo thiếu image_processor_type
    nsfw_clf = pipeline("image-classification", model=mdl, image_processor=ip,
                        device=0 if use_gpu else -1)
    log("[ok] NSFW:", mid, "| device:", "cuda:0" if use_gpu else "cpu")

    # --- OCR (RapidOCR / ONNXRuntime) ---
    try:
        from rapidocr import RapidOCR; _OCR_NEW = True
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR; _OCR_NEW = False
    ocr_engine = (_make_ocr_gpu(RapidOCR) if use_gpu else None) or RapidOCR()
    try:
        import onnxruntime as ort
        log("[ok] OCR: RapidOCR", "(new)" if _OCR_NEW else "(old)",
            "| ORT providers:", ort.get_available_providers())
    except Exception:
        log("[ok] OCR: RapidOCR", "(new)" if _OCR_NEW else "(old)")

    # --- WeChat QR (optional) ---
    try:
        m = PROJ / ".models" / "wechat_qrcode"; m.mkdir(parents=True, exist_ok=True)
        base = "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/"
        for f in ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"):
            if not (m / f).exists():
                log("    tải", f); urllib.request.urlretrieve(base + f, m / f)
        wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
            str(m / "detect.prototxt"), str(m / "detect.caffemodel"),
            str(m / "sr.prototxt"),     str(m / "sr.caffemodel"))
        log("[ok] WeChat QR")
    except Exception as e:
        wechat_detector = None
        log("[warn] WeChat QR off:", type(e).__name__, e)


# ======================================================================================
# I/O
# ======================================================================================
_HEADERS = {"User-Agent": "Mozilla/5.0"}

def load_image(src):
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if str(src).startswith("http"):
        import requests
        r = requests.get(src, headers=_HEADERS, timeout=30); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(src).convert("RGB")


# ======================================================================================
# 1) QR / BARCODE  (pyzbar đa scale+tiền xử lý -> cv2 -> WeChat)
# ======================================================================================
from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol

# Chỉ giải QR + barcode bán lẻ phổ biến. Bỏ PDF417/DATABAR (decoder libzbar hay assert -> spam
# warning + chậm, mà ta không cần). Lọc theo tên có thật để khỏi vỡ giữa các version pyzbar.
_ZSYM_NAMES = ["QRCODE", "EAN13", "EAN8", "UPCA", "UPCE",
               "CODE128", "CODE39", "CODE93", "I25", "CODABAR"]
_ZSYMS = [getattr(ZBarSymbol, n) for n in _ZSYM_NAMES if hasattr(ZBarSymbol, n)]

CODE_SCALES = (1.0, 1.5, 2.0, 3.0)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
_qr_cv = cv2.QRCodeDetector()

def _gray_variants(gray):
    yield gray
    yield _CLAHE.apply(gray)
    yield cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    yield cv2.filter2D(gray, -1, _SHARPEN)

def detect_codes(img):
    base = img.convert("RGB")
    found = {}
    for s in CODE_SCALES:
        im = base if s == 1.0 else base.resize((round(base.width * s), round(base.height * s)))
        for v in _gray_variants(np.array(im.convert("L"))):
            for d in zbar_decode(v, symbols=_ZSYMS):
                found[(d.type, bytes(d.data))] = None
        if found:
            break
    if not found:
        bgr = np.array(base)[:, :, ::-1].copy()
        try:
            ok, infos, _pts, _ = _qr_cv.detectAndDecodeMulti(bgr)
            if ok:
                for t in infos:
                    if t:
                        found[("QRCODE", t.encode())] = None
        except cv2.error:
            pass
    if not found and wechat_detector is not None:
        try:
            texts, _ = wechat_detector.detectAndDecode(np.array(base)[:, :, ::-1].copy())
            for t in texts:
                if t:
                    found[("QRCODE", t.encode())] = None
        except Exception:
            pass
    return [{"type": t, "data": data.decode("utf-8", "replace")} for (t, data) in found]


# ======================================================================================
# 2) NSFW
# ======================================================================================
UNSAFE_LABELS = {"porn", "hentai", "sexy"}
ALL_LABELS = ["neutral", "drawings", "sexy", "porn", "hentai"]

def nsfw_score(img):
    nb = {p["label"].lower(): float(p["score"]) for p in nsfw_clf(img)}
    return sum(nb.get(k, 0.0) for k in UNSAFE_LABELS)


# ======================================================================================
# 3) OCR + keyword
# ======================================================================================
def _parse_rapid(res):
    for tattr, sattr in (("txts", "scores"), ("rec_texts", "rec_scores")):
        txts = getattr(res, tattr, None)
        if txts is not None:
            scs = getattr(res, sattr, None) or [1.0] * len(txts)
            return [(t, float(s)) for t, s in zip(txts, scs)]
    result = res[0] if isinstance(res, tuple) else res
    out = []
    for item in (result or []):
        try:
            _, t, s = item; out.append((t, float(s)))
        except Exception:
            pass
    return out

def extract_text(img, min_conf=0.5):
    res = ocr_engine(np.array(img.convert("RGB")))
    return [(t, s) for (t, s) in _parse_rapid(res) if s >= min_conf]

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "@": "a", "!": "i", "$": "s", "|": "i"})
def _norm(s):
    s = unicodedata.normalize("NFKC", s).lower().translate(_LEET)
    return re.sub(r"[^a-z0-9]", "", s)

BANNED = {
    "tiktok": "tiktok", "tiktokshop": "tiktok", "douyin": "tiktok",
    "lazada": "competitor", "tiki": "competitor", "sendo": "competitor", "taobao": "competitor",
    "temu": "competitor", "aliexpress": "competitor", "1688": "competitor", "shein": "competitor",
    "facebook": "competitor", "instagram": "competitor", "youtube": "competitor",
    "telegram": "competitor", "zalo": "competitor", "wechat": "competitor", "kuaishou": "competitor",
}
_BANNED_NORM = {kn: v for k, v in BANNED.items() if (kn := _norm(k)) and len(kn) >= 4}
_RAW_KEYWORDS = {"抖音": "tiktok"}   # 抖音
_URL_RE = re.compile(r"(https?://|www\.|\.com|\.vn\b|@[a-z0-9_.]{3,})", re.I)

def text_violations(texts):
    hits = {}
    raw = " ".join(t for t, _ in texts)
    blob = _norm(raw)
    for kn, label in _BANNED_NORM.items():
        if kn in blob:
            hits.setdefault(label, []).append(kn)
    for kw, label in _RAW_KEYWORDS.items():
        if kw in raw:
            hits.setdefault(label, []).append(kw)
    for t, _ in texts:
        if _URL_RE.search(t):
            hits.setdefault("url", []).append(t.strip())
    return hits


# ======================================================================================
# PIPELINE
# ======================================================================================
def moderate(src, nsfw_thr=0.5):
    img = load_image(src)

    codes = detect_codes(img)
    if codes:
        return {"verdict": "REJECT", "violations": "qr_barcode",
                "nsfw": None, "detail": codes}

    nsfw = nsfw_score(img)
    if nsfw >= nsfw_thr:
        return {"verdict": "REJECT", "violations": "nsfw", "nsfw": round(nsfw, 3), "detail": None}

    hits = text_violations(extract_text(img))
    if hits:
        return {"verdict": "REJECT", "violations": ",".join(hits),
                "nsfw": round(nsfw, 3), "detail": hits}

    return {"verdict": "ACCEPT", "violations": "", "nsfw": round(nsfw, 3), "detail": None}


# ======================================================================================
# CLI
# ======================================================================================
_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def _collect(path):
    p = pathlib.Path(path)
    if p.is_dir():
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in _EXTS)
    return [str(p)]

def main():
    ap = argparse.ArgumentParser(description="Shopee image moderation (QR/NSFW/OCR)")
    ap.add_argument("path", help="ảnh hoặc folder ảnh")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="auto: dùng GPU nếu có (mặc định)")
    ap.add_argument("--workers", type=int, default=1, help="số thread chạy song song")
    ap.add_argument("--nsfw-thr", type=float, default=0.5)
    ap.add_argument("--csv", help="ghi kết quả ra file CSV")
    args = ap.parse_args()

    paths = _collect(args.path)
    if not paths:
        print("Không tìm thấy ảnh ở:", args.path); return

    # Quyết định CPU/GPU
    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass
    use_gpu = args.device == "gpu" or (args.device == "auto" and cuda_ok)
    if args.device == "gpu" and not cuda_ok:
        print("[warn] --device gpu nhưng torch không thấy CUDA -> chạy CPU")
        use_gpu = False

    print(f"Loading engines... ({len(paths)} ảnh, device={'gpu' if use_gpu else 'cpu'})")
    load_engines(use_gpu=use_gpu)

    if not use_gpu:
        try:
            import torch; torch.set_num_threads(1)           # CPU: để N worker không tranh hết core
        except Exception:
            pass

    def work(p):
        try:
            r = moderate(p, args.nsfw_thr)
            return p, r["verdict"], r["violations"], r["nsfw"]
        except Exception as e:
            return p, "ERROR", f"{type(e).__name__}: {e}", None

    t0 = time.time()
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(work, paths))
    else:
        results = [work(p) for p in paths]
    dt = time.time() - t0

    rej = 0
    for p, verdict, viol, nsfw in results:
        if verdict == "REJECT":
            rej += 1
        name = os.path.basename(p)
        print(f"{verdict:7} {name:48} {viol}")

    print(f"\n{len(paths)} ảnh / {dt:.1f}s ({dt/len(paths):.2f}s/ảnh, "
          f"device={'gpu' if use_gpu else 'cpu'}, workers={args.workers}) "
          f"| REJECT={rej} ACCEPT={len(paths)-rej}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "verdict", "violations", "nsfw"])
            for p, verdict, viol, nsfw in results:
                w.writerow([os.path.basename(p), verdict, viol, nsfw])
        print("CSV:", args.csv)


if __name__ == "__main__":
    main()
