#!/usr/bin/env python
"""Test API moderation trên cả folder ảnh, chạy song song.

    python test_api.py sample
    python test_api.py sample --url http://91.150.160.38:16815 --workers 8 --csv api_result.csv

Gửi từng ảnh tới POST /moderate của service rồi tổng hợp verdict + thời gian.
"""
import argparse, glob, os, time, csv
from concurrent.futures import ThreadPoolExecutor
import requests

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def moderate_one(url, path, nsfw_thr):
    t0 = time.time()
    try:
        with open(path, "rb") as f:
            r = requests.post(f"{url}/moderate",
                              files={"file": (os.path.basename(path), f)},
                              params={"nsfw_thr": nsfw_thr}, timeout=120)
        r.raise_for_status()
        d = r.json()
        return {"file": os.path.basename(path), "verdict": d.get("verdict"),
                "violations": d.get("violations", ""), "nsfw": d.get("nsfw"),
                "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"file": os.path.basename(path), "verdict": "ERROR",
                "violations": f"{type(e).__name__}: {e}", "nsfw": None,
                "ms": int((time.time() - t0) * 1000)}


def main():
    ap = argparse.ArgumentParser(description="Client test API moderation cả folder")
    ap.add_argument("folder", help="folder ảnh test")
    ap.add_argument("--url", default="http://91.150.160.38:16815", help="base URL của API")
    ap.add_argument("--workers", type=int, default=4, help="số request song song")
    ap.add_argument("--nsfw-thr", type=float, default=0.5)
    ap.add_argument("--csv", help="ghi kết quả ra CSV")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.folder, "*"))
                   if p.lower().endswith(_EXTS))
    if not paths:
        print("Không có ảnh trong", args.folder); return

    # Health check
    try:
        h = requests.get(f"{args.url}/health", timeout=10).json()
        print(f"API {args.url} | health={h}")
    except Exception as e:
        print(f"[warn] không gọi được /health: {e}")

    print(f"Gửi {len(paths)} ảnh, workers={args.workers} ...\n")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda p: moderate_one(args.url, p, args.nsfw_thr), paths))
    dt = time.time() - t0

    rej = err = 0
    for r in results:
        if r["verdict"] == "REJECT": rej += 1
        elif r["verdict"] == "ERROR": err += 1
        print(f"{r['verdict']:7} {r['file']:48} {r['violations']:20} {r['ms']:5}ms")

    print(f"\n{len(paths)} ảnh / {dt:.1f}s ({dt/len(paths):.3f}s/ảnh, workers={args.workers}) "
          f"| REJECT={rej} ACCEPT={len(paths)-rej-err} ERROR={err}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "verdict", "violations", "nsfw", "ms"])
            w.writeheader(); w.writerows(results)
        print("CSV:", args.csv)


if __name__ == "__main__":
    main()
