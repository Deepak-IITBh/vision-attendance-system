"""Measure OMR cell-classification accuracy against the synthetic ground truth.

Runs omr.analyze() (pure OpenCV path, no network) on every generated image and
compares the present/absent matrix to ground truth, reporting per-variant
accuracy and how the deterministic "confident" auto-return path performs.

    python tests/generate_sheets.py    # once, to create tests/_synth/
    python tests/eval_omr.py           # plain
    HINT=1 python tests/eval_omr.py    # also pass the true #students/#days

With the size hint the confident path is ~100% accurate; without it, a couple of
heavily-degraded sheets are mis-segmented but those are exactly the ones the
pipeline routes to the VLM fallback in production.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import omr  # noqa: E402

SYNTH = os.path.join(os.path.dirname(__file__), "_synth")


def score(pred, gt):
    correct = total = unclear = 0
    for r in range(len(gt)):
        for c in range(len(gt[0])):
            total += 1
            pv = pred[r][c] if r < len(pred) and c < len(pred[r]) else None
            if pv == "unclear":
                unclear += 1
            if pv == gt[r][c]:
                correct += 1
    return correct, total, unclear


def main():
    manifest = json.load(open(os.path.join(SYNTH, "manifest.json")))
    hint = os.environ.get("HINT") == "1"
    rows, by_variant = [], {}
    for item in manifest:
        gt = json.load(open(os.path.join(SYNTH, item["gt"])))
        data = open(os.path.join(SYNTH, item["image"]), "rb").read()
        kw = dict(allow_vlm=False)
        if hint:
            kw.update(expected_rows=gt["n_rows"], expected_cols=gt["n_cols"])
        try:
            res = omr.analyze(data, **kw)
        except Exception as exc:  # noqa: BLE001
            res = {"matrix": [], "error": repr(exc)}
        pred = res.get("matrix") or []
        correct, total, _ = score(pred, gt["matrix"])
        acc = correct / total if total else 0.0
        rows.append({"case": item["case"], "variant": item["variant"], "acc": acc,
                     "confident": res.get("confident"),
                     "shape_ok": len(pred) == gt["n_rows"] and
                     all(len(r) == gt["n_cols"] for r in pred) if pred else False})
        by_variant.setdefault(item["variant"], []).append(acc)

    print(f"{'case':16}{'variant':13}{'acc':>8}{'confident':>11}")
    for r in rows:
        print(f"{r['case']:16}{r['variant']:13}{r['acc']*100:7.1f}%{str(r['confident']):>11}")
    print("\nPer-variant mean accuracy:")
    for v, a in by_variant.items():
        print(f"  {v:14}{sum(a)/len(a)*100:6.2f}%")
    overall = [r["acc"] for r in rows]
    print(f"\nOverall: {sum(overall)/len(overall)*100:.2f}%   shape-correct {sum(r['shape_ok'] for r in rows)}/{len(rows)}")
    conf = [r for r in rows if r["confident"]]
    if conf:
        ca = [r["acc"] for r in conf]
        print(f"Confident auto-return: {len(conf)}/{len(rows)} cases @ {sum(ca)/len(ca)*100:.2f}% "
              f"(<97%: {sum(1 for r in conf if r['acc'] < 0.97)})")


if __name__ == "__main__":
    main()
