# OMR accuracy tests

Synthetic regression tests for the OpenCV OMR reader (`omr.py`). They render
attendance sheets with known ground truth, run the **pure OpenCV path**
(`allow_vlm=False`, no network), and measure cell-level present/absent accuracy.

```bash
python tests/generate_sheets.py    # creates tests/_synth/ (gitignored)
python tests/eval_omr.py           # measure accuracy
HINT=1 python tests/eval_omr.py    # also pass the true #students / #days
```

Coverage: 5 layouts (up to 20×30 = 600 cells) × 6 capture conditions
(clean, rotated, perspective "phone photo", blur+noise, low-res JPEG, combo).

What "good" looks like:

- **Clean / mild conditions:** ~100% cell accuracy on the confident path.
- **With the size hint** (`HINT=1`): the confident auto-return path is ~100%
  with **zero confidently-wrong sheets** — anything it can't read cleanly is
  marked low-confidence (in production that routes to the chunked VLM fallback).
- Heavily-degraded sheets may mis-segment without a hint; those are exactly the
  cases the app abstains on and falls back for, rather than returning wrong data.
