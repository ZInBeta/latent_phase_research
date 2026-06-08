import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-demos", type=str, required=True)
    parser.add_argument("--segments", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--view-size", type=int, default=256)
    parser.add_argument("--boundary-window", type=int, default=3)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_phase_at_frame(segments, frame):
    for s in segments:
        if int(s["start_frame"]) <= frame <= int(s["end_frame"]):
            return int(s["phase"])
    return None


def get_boundaries(segments):
    out = []
    for i in range(1, len(segments)):
        prev_s = segments[i - 1]
        cur_s = segments[i]
        out.append({
            "boundary_frame": int(cur_s["start_frame"]),
            "from_phase": int(prev_s["phase"]),
            "to_phase": int(cur_s["phase"]),
        })
    return out


def resize_rgb(arr, size):
    img = Image.fromarray(arr.astype(np.uint8))
    return img.resize((size, size))


def draw_frame(agent_img, wrist_img, frame_idx, phase, boundaries, boundary_window, view_size, title, font, small_font):
    top_h = 92
    gap = 12
    W = view_size * 2 + gap
    H = top_h + view_size

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    boundary_frames = [b["boundary_frame"] for b in boundaries]
    near = [b for b in boundaries if abs(frame_idx - b["boundary_frame"]) <= boundary_window]

    draw.text((8, 8), title, fill="black", font=small_font)
    draw.text((8, 30), f"frame = {frame_idx}    predicted phase = {phase}", fill="black", font=font)
    draw.text((8, 55), f"predicted boundaries = {boundary_frames}", fill="black", font=small_font)

    if near:
        info = ", ".join([f"{b['boundary_frame']} ({b['from_phase']}->{b['to_phase']})" for b in near])
        draw.text((8, 73), f"NEAR BOUNDARY: {info}", fill="red", font=small_font)

    canvas.paste(agent_img, (0, top_h))
    canvas.paste(wrist_img, (view_size + gap, top_h))

    draw.text((8, top_h + 5), "agentview", fill="white", font=small_font)
    draw.text((view_size + gap + 8, top_h + 5), "wrist", fill="white", font=small_font)

    if near:
        draw.rectangle((0, 0, W - 1, H - 1), outline="red", width=6)

    return np.array(canvas)


def write_video_cv2(frames, out_path, fps):
    import cv2

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open")

    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    writer.release()


def write_video_imageio(frames, out_path, fps):
    import imageio.v2 as imageio
    imageio.mimsave(str(out_path), frames, fps=fps)


def safe_name(file_name, demo_name):
    return f"{Path(file_name).stem}_{demo_name}".replace("/", "_")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.selected_demos, "r") as f:
        selected = json.load(f)

    seg_rows = load_jsonl(args.segments)
    seg_map = {
        (r["file_name"], r["demo_name"]): r["segments"]
        for r in seg_rows
    }

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
        small_font = None

    video_paths = []

    for item in selected:
        file_name = item["file_name"]
        demo_name = item["demo_name"]
        file_path = item["file_path"]

        key = (file_name, demo_name)
        if key not in seg_map:
            print("skip missing segments:", key)
            continue

        segments = seg_map[key]
        boundaries = get_boundaries(segments)

        with h5py.File(file_path, "r") as f:
            demo = f["data"][demo_name]
            obs = demo["obs"]
            agent = obs["agentview_rgb"][:]
            wrist = obs["eye_in_hand_rgb"][:]

        T = min(len(agent), len(wrist))
        frames = []

        title = f"{file_name} / {demo_name}"

        for t in range(T):
            phase = get_phase_at_frame(segments, t)
            agent_img = resize_rgb(agent[t], args.view_size)
            wrist_img = resize_rgb(wrist[t], args.view_size)

            fr = draw_frame(
                agent_img=agent_img,
                wrist_img=wrist_img,
                frame_idx=t,
                phase=phase,
                boundaries=boundaries,
                boundary_window=args.boundary_window,
                view_size=args.view_size,
                title=title,
                font=font,
                small_font=small_font,
            )
            frames.append(fr)

        out_path = out_dir / f"{safe_name(file_name, demo_name)}.mp4"

        try:
            write_video_cv2(frames, out_path, args.fps)
        except Exception as e:
            print("cv2 failed, trying imageio:", e)
            write_video_imageio(frames, out_path, args.fps)

        video_paths.append(out_path)
        print("saved:", out_path)

    # simple index html
    html = ["<html><head><meta charset='utf-8'><title>Semantic alignment videos</title></head><body>"]
    html.append("<h1>Semantic alignment videos</h1>")
    for p in video_paths:
        html.append(f"<h3>{p.name}</h3>")
        html.append(f"<video src='{p.name}' controls width='900'></video>")
    html.append("</body></html>")

    with open(out_dir / "video_index.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    zip_base = out_dir.parent / "semantic_alignment_videos"
    shutil.make_archive(str(zip_base), "zip", out_dir)

    print()
    print("Done.")
    print("video dir:", out_dir)
    print("video index:", out_dir / "video_index.html")
    print("zip:", out_dir / "semantic_alignment_videos.zip")


if __name__ == "__main__":
    main()
