#!/usr/bin/env python
"""Client: trích frame từ video (URL hoặc file) rồi gửi tới API /moderate, dừng sớm.

    python moderate_video_client.py "<video_url_hoặc_path>"
    python moderate_video_client.py video.mp4 --url http://IP:PORT --every-sec 1 --max-frames 60

Logic: lấy 1 frame mỗi --every-sec (dùng grab() bỏ qua frame thừa cho nhẹ CPU),
gửi từng frame tới /moderate; gặp frame REJECT đầu tiên -> kết luận video REJECT (kèm mốc giây).
"""
import argparse, time
import cv2
import requests


def moderate_video(video, api, every_sec, max_frames, nsfw_thr, verbose=True):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {"verdict": "ERROR", "reason": "không mở được video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_sec)), 1)

    idx = checked = 0
    t0 = time.time()
    while True:
        if not cap.grab():                       # rẻ: nhảy frame, không decode đầy đủ
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()           # chỉ decode frame cần
            if not ok:
                break
            tsec = idx / fps
            ok2, buf = cv2.imencode(".jpg", frame)
            try:
                r = requests.post(f"{api}/moderate",
                                  files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                                  params={"nsfw_thr": nsfw_thr}, timeout=60)
                d = r.json()
            except Exception as e:
                d = {"verdict": "ERROR", "violations": f"{type(e).__name__}: {e}"}
            checked += 1
            if verbose:
                print(f"  t={tsec:5.1f}s  {d.get('verdict'):7} {d.get('violations','')}")
            if d.get("verdict") == "REJECT":
                cap.release()
                return {"verdict": "REJECT", "at_sec": round(tsec, 1),
                        "violations": d.get("violations"), "frames_checked": checked,
                        "elapsed": round(time.time() - t0, 1)}
            if max_frames and checked >= max_frames:
                break
        idx += 1
    cap.release()
    return {"verdict": "ACCEPT", "frames_checked": checked,
            "elapsed": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description="Trích frame video -> gọi API /moderate")
    ap.add_argument("video", help="URL hoặc đường dẫn file video")
    ap.add_argument("--url", default="http://91.150.160.38:16815", help="base URL API")
    ap.add_argument("--every-sec", type=float, default=1.0, help="lấy 1 frame mỗi N giây")
    ap.add_argument("--max-frames", type=int, default=120, help="cap số frame mỗi video")
    ap.add_argument("--nsfw-thr", type=float, default=0.5)
    args = ap.parse_args()

    print(f"Video: {args.video}\nLấy 1 frame / {args.every_sec}s, gửi tới {args.url} ...")
    res = moderate_video(args.video, args.url, args.every_sec, args.max_frames, args.nsfw_thr)
    print("\n=> KẾT QUẢ:", res)


if __name__ == "__main__":
    main()
