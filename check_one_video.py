#!/usr/bin/env python
"""Check 1 video lẻ (URL hoặc file) qua API /moderate — in chi tiết TỪNG frame.

    python check_one_video.py "https://...mp4"
    python check_one_video.py video.mp4 --frames 12 --url http://IP:PORT --nsfw-thr 0.7

Quét N frame đều theo thời lượng, gọi /moderate cho từng frame (KHÔNG dừng sớm) để thấy
toàn cảnh, rồi tổng hợp: mỗi loại vi phạm dính bao nhiêu frame + điểm nsfw cao nhất.
"""
import argparse, sys
from collections import Counter
import cv2
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description="Check 1 video lẻ qua API")
    ap.add_argument("video", help="URL hoặc đường dẫn file video")
    ap.add_argument("--url", default="http://91.150.160.38:16815", help="base URL API")
    ap.add_argument("--frames", type=int, default=10, help="số frame lấy đều theo thời lượng")
    ap.add_argument("--nsfw-thr", type=float, default=0.7)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("✗ Không mở được video:", args.video); return
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = total / fps if fps else 0
    print(f"Video: {int(total)} frame | {dur:.1f}s @ {fps:.0f}fps | lấy {args.frames} frame, "
          f"nsfw_thr={args.nsfw_thr}\n")

    n = max(args.frames, 1)
    fracs = [(i + 0.5) / n for i in range(n)] if total > 0 else [0.0]

    cnt = Counter()
    max_nsfw = 0.0
    checked = 0
    for f in fracs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f * total))
        ok, frame = cap.read()
        if not ok:
            continue
        ok2, buf = cv2.imencode(".jpg", frame)
        try:
            r = requests.post(f"{args.url}/moderate", params={"nsfw_thr": args.nsfw_thr},
                              files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")}, timeout=60)
            d = r.json()
        except Exception as e:
            d = {"verdict": "ERROR", "violations": f"{type(e).__name__}: {e}", "nsfw": None}
        checked += 1
        v, viol, nsfw = d.get("verdict"), d.get("violations", ""), d.get("nsfw")
        nsfw_s = f"{nsfw:.2f}" if isinstance(nsfw, (int, float)) else "  - "
        print(f"  t={f*dur:5.1f}s ({f*100:3.0f}%)  {v:7} nsfw={nsfw_s}  {viol}")
        if v == "REJECT":
            for x in viol.split(","):
                if x:
                    cnt[x] += 1
        if isinstance(nsfw, (int, float)):
            max_nsfw = max(max_nsfw, nsfw)
    cap.release()

    verdict = "REJECT" if cnt else "ACCEPT"
    detail = " | ".join(f"{k}: {v}/{checked} frame" for k, v in cnt.items()) or "-"
    print(f"\n=> {verdict}  | {detail}  | nsfw cao nhất = {max_nsfw:.2f}")


if __name__ == "__main__":
    main()
