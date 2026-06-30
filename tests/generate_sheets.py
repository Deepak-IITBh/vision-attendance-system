"""Generate synthetic attendance sheets with known ground truth.

Renders bordered grids of roll numbers x date columns where each data cell is a
FILLED circle (present) or an EMPTY circle (absent), matching the real sample
format, plus degraded variants (rotation, perspective "phone photo", blur+noise,
low-res JPEG) to stress the OMR pipeline. Output goes to tests/_synth/ alongside
a per-image ground-truth JSON. Run, then evaluate with tests/eval_omr.py.

    python tests/generate_sheets.py
"""

import json
import os
import random

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "_synth")
os.makedirs(OUT, exist_ok=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_sheet(rolls, dates, matrix, *, style="printed", seed=0):
    rng = random.Random(seed)
    n_rows, n_cols = len(rolls), len(dates)
    id_w, col_w, header_h, row_h, pad, title_h = 150, 110, 70, 70, 40, 70
    W = pad * 2 + id_w + col_w * n_cols
    H = pad * 2 + title_h + header_h + row_h * n_rows
    img = np.full((H, W, 3), 250, np.uint8)
    ink = (30, 30, 30) if style == "printed" else (150, 70, 30)
    x0, y0 = pad, pad + title_h
    grid_w, grid_h = id_w + col_w * n_cols, header_h + row_h * n_rows

    cv2.putText(img, "Class Attendance", (pad + 10, pad + 45), FONT, 1.1, ink, 2, cv2.LINE_AA)
    xs = [x0, x0 + id_w] + [x0 + id_w + col_w * (c + 1) for c in range(n_cols)]
    ys = [y0, y0 + header_h] + [y0 + header_h + row_h * (r + 1) for r in range(n_rows)]
    for x in xs:
        cv2.line(img, (x, y0), (x, y0 + grid_h), ink, 2, cv2.LINE_AA)
    for y in ys:
        cv2.line(img, (x0, y), (x0 + grid_w, y), ink, 2, cv2.LINE_AA)

    cv2.putText(img, "Roll No", (x0 + 12, y0 + 45), FONT, 0.7, ink, 2, cv2.LINE_AA)
    for c, d in enumerate(dates):
        cv2.putText(img, str(d), (x0 + id_w + col_w * c + 12, y0 + 45), FONT, 0.6, ink, 2, cv2.LINE_AA)

    for r, roll in enumerate(rolls):
        ry = y0 + header_h + row_h * r
        cv2.putText(img, str(roll), (x0 + 14, ry + 45), FONT, 0.75, ink, 2, cv2.LINE_AA)
        for c in range(n_cols):
            cx = x0 + id_w + col_w * c + col_w // 2 + rng.randint(-6, 6)
            cy = ry + row_h // 2 + rng.randint(-5, 5)
            rad = rng.randint(18, 23)
            if matrix[r][c] == "present":
                if style == "printed":
                    cv2.circle(img, (cx, cy), rad, ink, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (cx, cy), rad, ink, 2, cv2.LINE_AA)
                    for t in range(-rad, rad, 4):
                        cv2.line(img, (cx - rad, cy + t), (cx + rad, cy + t), ink, 2, cv2.LINE_AA)
            else:
                cv2.circle(img, (cx, cy), rad, ink, 2, cv2.LINE_AA)
    return img


def degrade(img, kind, seed=0):
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    if kind == "clean":
        return img
    if kind == "rotate":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), 1.0)
        return cv2.warpAffine(img, m, (w, h), borderValue=(250, 250, 250))
    if kind == "perspective":
        bg = np.full((int(h * 1.4), int(w * 1.4), 3), 0, np.uint8)
        bg[:] = (120, 160, 190)
        bg = np.clip(bg.astype(np.int16) + rng.randint(-12, 12, bg.shape), 0, 255).astype(np.uint8)
        oy, ox = (bg.shape[0] - h) // 2, (bg.shape[1] - w) // 2
        bg[oy:oy + h, ox:ox + w] = img
        d = 0.06
        src = np.float32([[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]])
        dst = np.float32([
            [ox + rng.uniform(0, d) * w, oy + rng.uniform(0, d) * h],
            [ox + w - rng.uniform(0, d) * w, oy + rng.uniform(0, d * 1.5) * h],
            [ox + w - rng.uniform(0, d * 1.5) * w, oy + h - rng.uniform(0, d) * h],
            [ox + rng.uniform(0, d) * w, oy + h - rng.uniform(0, d) * h],
        ])
        mt = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(bg, mt, (bg.shape[1], bg.shape[0]), borderValue=(120, 160, 190))
    if kind == "blurnoise":
        out = cv2.GaussianBlur(img, (0, 0), 1.2)
        return np.clip(out.astype(np.int16) + rng.randint(-18, 18, out.shape), 0, 255).astype(np.uint8)
    if kind == "lowres_jpeg":
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 45])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "combo":
        return degrade(degrade(img, "rotate", seed), "blurnoise", seed + 1)
    raise ValueError(kind)


def make_case(name, n_rows, n_cols, style, seed, present_rate):
    rng = random.Random(seed)
    rolls = [f"R{100 + i + 1}" for i in range(n_rows)]
    dates = [f"D{c + 1:02d}" for c in range(n_cols)]
    matrix = [["present" if rng.random() < present_rate else "absent"
               for _ in range(n_cols)] for _ in range(n_rows)]
    img = render_sheet(rolls, dates, matrix, style=style, seed=seed)
    return img, {"name": name, "rolls": rolls, "dates": dates, "matrix": matrix,
                 "n_rows": n_rows, "n_cols": n_cols}


def main():
    cases = [
        ("small_printed", 8, 3, "printed", 0.6),
        ("medium_printed", 12, 10, "printed", 0.65),
        ("large_printed", 20, 30, "printed", 0.7),
        ("large_scribble", 20, 30, "scribble", 0.7),
        ("tall_printed", 25, 6, "printed", 0.55),
    ]
    variants = ["clean", "rotate", "perspective", "blurnoise", "lowres_jpeg", "combo"]
    manifest = []
    for i, (name, nr, nc, style, pr) in enumerate(cases):
        base, gt = make_case(name, nr, nc, style, i * 7 + 1, pr)
        for v in variants:
            img = degrade(base.copy(), v, seed=i * 13 + 3)
            cv2.imwrite(os.path.join(OUT, f"{name}__{v}.png"), img)
            with open(os.path.join(OUT, f"{name}__{v}.json"), "w") as f:
                json.dump(gt, f)
            manifest.append({"image": f"{name}__{v}.png", "gt": f"{name}__{v}.json",
                             "case": name, "variant": v})
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} images to {OUT}")


if __name__ == "__main__":
    main()
