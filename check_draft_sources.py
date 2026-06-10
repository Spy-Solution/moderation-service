#!/usr/bin/env python
"""Quét bảng draft_source (status='ready'), kiểm duyệt video, cảnh báo Discord nếu vi phạm.

Mỗi video: lấy 5 frame ở các vị trí [0.1, 0.25, 0.5, 0.75, 0.9] theo thời lượng -> gửi API
/moderate. Bất kỳ frame nào REJECT -> video REJECT -> warning Discord. ACCEPT thì bỏ qua.

Cấu hình trong .env:
    DATABASE_URL          postgresql://...
    DISCORD_TOKEN         bot token
    DISCORD_CHANNEL       channel id
    MODERATION_API_URL    (tuỳ chọn) base URL API, mặc định endpoint Vast.ai

Chạy:
    python check_draft_sources.py                 # quét hết, gửi Discord thật
    python check_draft_sources.py --dry-run       # không gửi Discord, chỉ in
    python check_draft_sources.py --limit 5       # chỉ 5 video đầu (test)
"""
import argparse, os, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
DISCORD_CHANNEL = os.environ["DISCORD_CHANNEL"]
API_URL = os.environ.get("MODERATION_API_URL", "http://91.150.160.38:16815")

POSITIONS = [0.1, 0.25, 0.5, 0.75, 0.9]
QUERY = "SELECT * FROM public.draft_source WHERE status = 'ready' ORDER BY id ASC"


def frames_at(url):
    """Lấy frame ở các vị trí POSITIONS. Trả về list (frac, frame) hoặc None nếu mở fail."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    out = []
    if total <= 0:                       # không biết tổng frame -> đọc tuần tự lấy frame đầu
        ok, frame = cap.read()
        if ok:
            out.append((0.0, frame))
    else:
        for f in POSITIONS:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f * total))
            ok, frame = cap.read()
            if ok:
                out.append((f, frame))
    cap.release()
    return out


def moderate_frame(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    r = requests.post(f"{API_URL}/moderate",
                      files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}, timeout=60)
    r.raise_for_status()
    return r.json()


def check_video(url):
    """Trả về dict verdict. REJECT ngay khi 1 frame vi phạm (kèm vị trí + lý do)."""
    frames = frames_at(url)
    if frames is None:
        return {"verdict": "ERROR", "reason": "không mở được video"}
    if not frames:
        return {"verdict": "ERROR", "reason": "không đọc được frame"}
    for frac, frame in frames:
        try:
            d = moderate_frame(frame)
        except Exception as e:
            return {"verdict": "ERROR", "reason": f"API: {type(e).__name__}: {e}"}
        if d.get("verdict") == "REJECT":
            return {"verdict": "REJECT", "at": frac, "violations": d.get("violations"),
                    "frames_checked": len(frames)}
    return {"verdict": "ACCEPT", "frames_checked": len(frames)}


def discord_warn(content):
    r = requests.post(f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages",
                      headers={"Authorization": f"Bot {DISCORD_TOKEN}"},
                      json={"content": content}, timeout=30)
    if r.status_code >= 300:
        print(f"  [discord lỗi {r.status_code}] {r.text[:200]}")
    return r.status_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="không gửi Discord, chỉ in")
    ap.add_argument("--limit", type=int, help="chỉ xử lý N video đầu")
    args = ap.parse_args()

    # health check API
    try:
        h = requests.get(f"{API_URL}/health", timeout=10).json()
        print(f"API {API_URL} | {h}")
    except Exception as e:
        print(f"[FATAL] API không phản hồi: {e}"); return

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    q = QUERY + (f" LIMIT {int(args.limit)}" if args.limit else "")
    cur.execute(q)
    cols = [c[0] for c in cur.description]
    url_i, id_i = cols.index("url"), (cols.index("id") if "id" in cols else 0)
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"{len(rows)} video status='ready'\n")

    rej = err = 0
    t0 = time.time()
    for row in rows:
        vid, url = row[id_i], row[url_i]
        if not url:
            print(f"id={vid} [skip] url rỗng"); continue
        res = check_video(url)
        v = res["verdict"]
        if v == "REJECT":
            rej += 1
            at = f"{res['at']*100:.0f}%"
            msg = (f"⚠️ **Video vi phạm** (id=`{vid}`)\n"
                   f"Lý do: **{res['violations']}** tại ~{at} thời lượng\n{url}")
            print(f"id={vid} REJECT  {res['violations']} @ {at}")
            if not args.dry_run:
                discord_warn(msg)
        elif v == "ERROR":
            err += 1
            print(f"id={vid} ERROR   {res['reason']}")
        else:
            print(f"id={vid} ACCEPT")

    print(f"\nXong {len(rows)} video / {time.time()-t0:.1f}s "
          f"| REJECT={rej} ACCEPT={len(rows)-rej-err} ERROR={err}"
          + (" (dry-run, không gửi Discord)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
