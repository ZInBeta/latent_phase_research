import argparse
import base64
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/outputs/demo_split_test_phase_eval/plots",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/outputs/demo_split_test_phase_eval/phase_viz_report.html",
    )
    parser.add_argument("--pattern", type=str, default="*_combined.png")
    return parser.parse_args()


def image_to_base64(path):
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def main():
    args = parse_args()
    plot_dir = Path(args.plot_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(plot_dir.rglob(args.pattern))

    if not images:
        raise FileNotFoundError(f"No images found in {plot_dir} with pattern {args.pattern}")

    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html>")
    html.append("<head>")
    html.append("<meta charset='utf-8'>")
    html.append("<title>Phase-Belief Test Demo Visualization Report</title>")
    html.append("""
<style>
body {
  font-family: Arial, sans-serif;
  margin: 32px;
  background: #f7f7f7;
  color: #222;
}
h1 {
  margin-bottom: 8px;
}
.task {
  margin-top: 36px;
  padding-top: 16px;
  border-top: 2px solid #ddd;
}
.figure {
  background: white;
  padding: 16px;
  margin: 20px 0;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.figure h3 {
  margin-top: 0;
  font-size: 16px;
}
img {
  width: 100%;
  max-width: 1400px;
  display: block;
}
.meta {
  color: #666;
  margin-bottom: 24px;
}
</style>
""")
    html.append("</head>")
    html.append("<body>")
    html.append("<h1>Phase-Belief Test Demo Visualization Report</h1>")
    html.append(f"<div class='meta'>Source directory: {plot_dir}<br>Total images: {len(images)}</div>")

    current_task = None

    for img_path in images:
        rel = img_path.relative_to(plot_dir)
        task = rel.parts[0] if len(rel.parts) > 1 else "root"

        if task != current_task:
            current_task = task
            html.append(f"<div class='task'><h2>{task}</h2></div>")

        b64 = image_to_base64(img_path)
        html.append("<div class='figure'>")
        html.append(f"<h3>{rel}</h3>")
        html.append(f"<img src='data:image/png;base64,{b64}' />")
        html.append("</div>")

    html.append("</body>")
    html.append("</html>")

    out_path.write_text("\n".join(html), encoding="utf-8")

    print(f"saved: {out_path}")
    print(f"num images: {len(images)}")


if __name__ == "__main__":
    main()
