import argparse
import base64
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-records", type=str, required=True)
    parser.add_argument("--segments", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--demos-per-task", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=36)
    parser.add_argument("--boundary-context", type=int, default=5)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def demo_sort_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def get_phase_at_frame(segments, frame):
    for s in segments:
        if s["start_frame"] <= frame <= s["end_frame"]:
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
            "prev_len": int(prev_s["length"]),
            "next_len": int(cur_s["length"]),
        })
    return out


def select_frames(T, boundaries, max_frames, context):
    frames = set()

    if T <= max_frames:
        return list(range(T))

    uniform = np.linspace(0, T - 1, max_frames // 2).round().astype(int).tolist()
    frames.update(uniform)

    for b in boundaries:
        f = b["boundary_frame"]
        for delta in [-context, 0, context]:
            ff = int(np.clip(f + delta, 0, T - 1))
            frames.add(ff)

    frames = sorted(frames)

    if len(frames) > max_frames:
        boundary_frames = set()
        for b in boundaries:
            f = b["boundary_frame"]
            for delta in [-context, 0, context]:
                boundary_frames.add(int(np.clip(f + delta, 0, T - 1)))

        keep = sorted(boundary_frames)
        remain = [f for f in frames if f not in boundary_frames]
        need = max_frames - len(keep)

        if need > 0 and remain:
            idx = np.linspace(0, len(remain) - 1, min(need, len(remain))).round().astype(int)
            keep.extend([remain[i] for i in idx])

        frames = sorted(set(keep))[:max_frames]

    return frames


def resize_img(arr, size=(128, 128)):
    img = Image.fromarray(arr.astype(np.uint8))
    return img.resize(size)


def make_contact_sheet(agent, wrist, frames, segments, boundaries, out_path, title):
    tile_w = 280
    tile_h = 170
    cols = 4
    rows = int(np.ceil(len(frames) / cols))

    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h + 40), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = None
        title_font = None

    draw.text((10, 10), title, fill="black", font=title_font)

    boundary_frames = [b["boundary_frame"] for b in boundaries]

    for idx, f in enumerate(frames):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = r * tile_h + 40

        ag = resize_img(agent[f])
        wr = resize_img(wrist[f])

        canvas = Image.new("RGB", (tile_w, tile_h), "white")
        canvas.paste(ag, (8, 24))
        canvas.paste(wr, (144, 24))

        phase = get_phase_at_frame(segments, f)
        near_boundary = any(abs(f - bf) <= 2 for bf in boundary_frames)

        label = f"frame={f} phase={phase}"
        if near_boundary:
            label += "  boundary±2"

        d = ImageDraw.Draw(canvas)
        d.text((8, 5), label, fill="red" if near_boundary else "black", font=font)
        d.text((8, 154), "agentview", fill="black", font=font)
        d.text((144, 154), "wrist", fill="black", font=font)

        if near_boundary:
            d.rectangle((2, 2, tile_w - 3, tile_h - 3), outline="red", width=3)

        sheet.paste(canvas, (x0, y0))

    sheet.save(out_path)


def make_timeline_plot(actions, gripper, segments, boundaries, out_path, title):
    T = len(actions)
    phase_per_frame = np.full(T, np.nan, dtype=np.float32)

    for s in segments:
        phase_per_frame[s["start_frame"]:s["end_frame"] + 1] = int(s["phase"])

    action_norm = np.linalg.norm(actions[:, :6], axis=-1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)

    axes[0].plot(np.arange(T), phase_per_frame)
    axes[0].set_ylabel("argmax phase")
    axes[0].set_title(title)

    axes[1].plot(np.arange(T), action_norm)
    axes[1].set_ylabel("action norm")

    axes[2].plot(np.arange(T), gripper[:, 0], label="gripper 0")
    if gripper.shape[1] > 1:
        axes[2].plot(np.arange(T), gripper[:, 1], label="gripper 1")
    axes[2].set_ylabel("gripper")
    axes[2].set_xlabel("frame")
    axes[2].legend()

    for ax in axes:
        for b in boundaries:
            ax.axvline(b["boundary_frame"], linestyle="--", linewidth=1)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def img_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_records, "r") as f:
        split_records = json.load(f)

    records = split_records[args.split]
    records = sorted(records, key=lambda r: (r["file_name"], demo_sort_key(r["demo_name"])))

    seg_rows = load_jsonl(args.segments)
    seg_map = {(r["file_name"], r["demo_name"]): r["segments"] for r in seg_rows}

    selected = []
    count_by_task = {}

    for r in records:
        fn = r["file_name"]
        count_by_task.setdefault(fn, 0)

        if count_by_task[fn] >= args.demos_per_task:
            continue

        key = (r["file_name"], r["demo_name"])
        if key not in seg_map:
            continue

        selected.append(r)
        count_by_task[fn] += 1

    annotation_rows = []
    boundary_rows = []
    selected_rows = []

    html = []
    html.append("<html><head><meta charset='utf-8'><title>Semantic Alignment Samples</title>")
    html.append("""
<style>
body { font-family: Arial, sans-serif; margin: 28px; background: #f6f6f6; }
.demo { background: white; padding: 18px; margin: 24px 0; border-radius: 10px; }
img { max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }
pre { background: #f0f0f0; padding: 10px; overflow-x: auto; }
</style>
""")
    html.append("</head><body>")
    html.append("<h1>Semantic Alignment Samples</h1>")
    html.append("<p>Annotate event frames in semantic_annotation_template.csv. Boundary frames are predicted phase transitions.</p>")

    event_names = [
        "event_1_approach_or_search",
        "event_2_contact_grasp_or_open",
        "event_3_transport_or_manipulate",
        "event_4_place_release_or_finish",
    ]

    for r in selected:
        file_path = r["file_path"]
        file_name = r["file_name"]
        demo_name = r["demo_name"]
        segments = seg_map[(file_name, demo_name)]
        boundaries = get_boundaries(segments)

        with h5py.File(file_path, "r") as f:
            demo = f["data"][demo_name]
            obs = demo["obs"]
            agent = obs["agentview_rgb"][:]
            wrist = obs["eye_in_hand_rgb"][:]
            actions = demo["actions"][:]
            gripper = obs["gripper_states"][:]

        T = len(actions)
        frames = select_frames(T, boundaries, args.max_frames, args.boundary_context)

        stem = f"{Path(file_name).stem}_{demo_name}"
        contact_path = img_dir / f"{stem}_contact_sheet.png"
        timeline_path = img_dir / f"{stem}_timeline.png"

        title = f"{file_name} / {demo_name}"
        make_contact_sheet(agent, wrist, frames, segments, boundaries, contact_path, title)
        make_timeline_plot(actions, gripper, segments, boundaries, timeline_path, title)

        selected_rows.append({
            "file_name": file_name,
            "demo_name": demo_name,
            "file_path": file_path,
            "num_frames": T,
            "num_boundaries": len(boundaries),
            "contact_sheet": str(contact_path),
            "timeline": str(timeline_path),
        })

        for b in boundaries:
            row = {
                "file_name": file_name,
                "demo_name": demo_name,
                **b,
            }
            boundary_rows.append(row)

        for ev in event_names:
            annotation_rows.append({
                "file_name": file_name,
                "demo_name": demo_name,
                "event_name": ev,
                "event_frame": "",
                "notes": "",
            })

        html.append("<div class='demo'>")
        html.append(f"<h2>{file_name} / {demo_name}</h2>")
        html.append(f"<p>Frames: {T}. Predicted boundaries: {[b['boundary_frame'] for b in boundaries]}</p>")
        html.append("<h3>Timeline</h3>")
        html.append(f"<img src='data:image/png;base64,{img_to_base64(timeline_path)}'>")
        html.append("<h3>Contact sheet</h3>")
        html.append(f"<img src='data:image/png;base64,{img_to_base64(contact_path)}'>")
        html.append("<h3>Predicted boundaries</h3>")
        html.append("<pre>")
        for b in boundaries:
            html.append(json.dumps(b, ensure_ascii=False))
        html.append("</pre>")
        html.append("</div>")

    html.append("</body></html>")

    with open(out_dir / "semantic_alignment_samples.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    with open(out_dir / "semantic_annotation_template.csv", "w", newline="") as f:
        fieldnames = ["file_name", "demo_name", "event_name", "event_frame", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotation_rows)

    with open(out_dir / "predicted_boundaries.csv", "w", newline="") as f:
        fieldnames = ["file_name", "demo_name", "boundary_frame", "from_phase", "to_phase", "prev_len", "next_len"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(boundary_rows)

    with open(out_dir / "selected_demos.json", "w") as f:
        json.dump(selected_rows, f, indent=2)

    print("Done.")
    print("Selected demos:", len(selected))
    print("HTML:", out_dir / "semantic_alignment_samples.html")
    print("Annotation template:", out_dir / "semantic_annotation_template.csv")
    print("Predicted boundaries:", out_dir / "predicted_boundaries.csv")


if __name__ == "__main__":
    main()
