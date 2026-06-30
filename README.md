# 📋 Attendance Detector

<<<<<<< HEAD
Upload a photo or scan of a class **attendance sheet** and get back a clean
present/absent table, keyed by **roll number**. The marks are read
**deterministically with OpenCV** (optical mark recognition), so the result does
not hallucinate and stays accurate even on large sheets (e.g. 30 days × 20+
students). A vision model is used only for small, reliable sub-tasks.

How a sheet is read ([`omr.py`](omr.py)):

1. **Rectify** — normalize resolution, perspective-warp a tilted phone photo to a
   flat rectangle, then deskew any rotation using the table's own rule lines.
2. **Detect the grid** — find the horizontal/vertical rules and rebuild the
   matrix of cells. Thresholds are relative to the *table* (not the image), so it
   handles merged header banners (e.g. a “JUNE (1–30)” cell spanning all day
   columns) and dense hairline grids without collapsing columns.
3. **Assign roles** — work out which leftmost column is the **roll number** and
   which columns are the date marks; header / title / legend bands are dropped.
4. **Read each circle (OMR)** — measure the *interior fill* of every date cell: a
   **filled / shaded / scribbled circle = present**, an **open circle = absent**.
   A per-sheet adaptive threshold (k-means over all fill ratios) handles printed
   black vs blue-pen vs pencil, and genuinely ambiguous cells are flagged
   **unclear** instead of guessed.
5. **Score confidence & fall back** — if the grid is clean and the marks separate
   cleanly, the OpenCV result is returned directly (**no AI call for the marks**).
   If the sheet is too blurry / skewed to trust, it falls back to the vision
   model but reads the sheet **in small tiles** (never the whole grid at once),
   which is what previously caused hallucination on big sheets.

The vision model only ever does two small jobs: OCR the roll-number column from a
narrow crop, and the tiled fallback above. The output never includes student
names — only the **roll number** and the per-day present/absent state.

> Flask backend · OpenCV OMR engine · plain HTML/CSS/JS frontend · deploys to Render.
=======
Application Demo Link: https://vision-attendance-system.onrender.com

Upload a photo or scan of a class **attendance sheet** and get back a clean
present/absent table. The app:

1. **Preprocesses** the image with OpenCV (resize → denoise → contrast boost →
   auto-deskew → sharpen, with an optional high-contrast binarize) so faint or
   blurry circles become readable.
2. Sends the cleaned image to a **Hugging Face vision-language model** which
   reads the names, dates, and circles — a **filled circle = present**, an
   **empty circle = absent** — and returns structured JSON.
3. Shows the result in a color-coded table with present/absent counts.

> Flask backend · plain HTML/CSS/JS frontend · deploys to Render.
>>>>>>> 

---

## Project structure

```
Attendance_detector/
├── app.py              # Flask app + API routes
<<<<<<< HEAD
├── omr.py              # OpenCV OMR engine (rectify, grid, roles, mark reading, fallback)
├── preprocess.py       # OpenCV image cleanup (preview + shared image helpers)
├── hf_client.py        # Hugging Face client (roll-strip OCR + tiled chunk reads)
=======
├── preprocess.py       # OpenCV image cleanup
├── hf_client.py        # Hugging Face Inference Providers client
>>>>>>> a062e5639debf1cb57716998e5203e220042c921
├── requirements.txt
├── render.yaml         # Render blueprint
├── Procfile
├── runtime.txt         # pins Python 3.12 for stable wheels
├── .env.example
├── templates/
│   └── index.html
<<<<<<< HEAD
├── static/
│   ├── style.css
│   └── script.js
└── tests/              # synthetic accuracy harness (see tests/README.md)
    ├── generate_sheets.py
    └── eval_omr.py
=======
└── static/
    ├── style.css
    └── script.js

```

---

<<<<<<< HEAD
## 1. Hugging Face token (optional)

The present/absent marks are read **entirely offline with OpenCV**, so a token is
**not required**. Set one only to (a) OCR real roll numbers from the sheet and
(b) enable the AI fallback for hard photos. Without a token, roll numbers fall
back to row numbers (1, 2, 3, …) and the fallback is disabled.

1. Sign in at <https://huggingface.co> → **Settings → Access Tokens**
   (<https://huggingface.co/settings/tokens>) → create a **Read** token (`hf_…`).
2. The default model is `Qwen/Qwen3-VL-8B-Instruct`, served via HF **Inference
   Providers** (a free account includes a monthly credit). A model only works if
   a provider you've enabled serves it — manage providers at
   <https://huggingface.co/settings/inference-providers>. On a "not supported by
   any provider" error, switch `HF_MODEL` (see Configuration).
=======
## 1. Get a Hugging Face token (free)

1. Sign in / create an account at <https://huggingface.co>.
2. Go to **Settings → Access Tokens** → <https://huggingface.co/settings/tokens>.
3. Create a token with **Read** access and copy it (`hf_...`).

The default model is `Qwen/Qwen3-VL-8B-Instruct`, served through HF
**Inference Providers**. A free account includes a monthly inference credit. A
model only works if a provider you've enabled serves it — check/enable providers
at <https://huggingface.co/settings/inference-providers>. If you hit a "not
supported by any provider you have enabled" error, switch `HF_MODEL` (see below).
>>>>>>

---

## 2. Run locally

```bash
# from the Attendance_detector folder
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

<<<<<<< HEAD
cp .env.example .env          # optional: paste your HF_TOKEN
python app.py
```

Open <http://localhost:5000>, upload a sheet, optionally click **Preview filter**
to see the cleaned image, then **Analyze attendance**.

Health check: <http://localhost:5000/healthz> (shows whether a token is set).
=======
cp .env.example .env          # then edit .env and paste your HF_TOKEN
python app.py
```

Open <http://localhost:5000>, upload a sheet, click **Preview filter** to see the
cleaned image, then **Analyze attendance**.

Health check: <http://localhost:5000/healthz> (shows whether the token is set).
>>>>>>> 

---

## 3. Deploy to Render

### Option A — Blueprint (recommended)
1. Push this folder to a GitHub repo.
2. In Render: **New → Blueprint**, pick the repo. Render reads `render.yaml`.
<<<<<<< HEAD
3. (Optional) set the **`HF_TOKEN`** environment variable (marked `sync: false`,
   so it is never stored in the repo).
=======
3. When prompted, set the **`HF_TOKEN`** environment variable to your token
   (it's marked `sync: false`, so it is never stored in the repo).

4. Deploy. The public URL serves the same UI.

### Option B — Manual Web Service
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT`
<<<<<<< HEAD
- **Environment variables:** `HF_MODEL` (optional), `HF_TOKEN` (optional),
  `PYTHON_VERSION=3.12.7`

> Keep the gunicorn `--timeout` at 120s: the tiled VLM fallback is bounded to a
> ~90s budget and returns a partial result before the worker is killed. On
> Render's **free** plan the service sleeps after inactivity, so the first
=======
- **Environment variables:** `HF_TOKEN` (required), `HF_MODEL` (optional),
  `PYTHON_VERSION=3.12.7`

> The long gunicorn `--timeout` matters: vision-model calls take ~10–30s and the
> default 30s timeout would otherwise kill the request.
>
> On Render's **free** plan the service sleeps after inactivity, so the first
>>>>>>> a062e5639debf1cb57716998e5203e220042c921
> request after idle may take a while to wake up.

---

## Configuration

<<<<<<< HEAD
| Variable   | Required | Default                       | Notes |
|------------|----------|-------------------------------|-------|
| `HF_TOKEN` | no       | —                             | Hugging Face token. Marks are read offline regardless; the token only enables roll-number OCR and the hard-photo fallback. |
| `HF_MODEL` | no       | `Qwen/Qwen3-VL-8B-Instruct`   | Any **image-capable** model served by a provider you've enabled. |

Other vision models that work well: `Qwen/Qwen3-VL-30B-A3B-Instruct`,
`Qwen/Qwen2.5-VL-72B-Instruct`, `google/gemma-3-27b-it`,
`meta-llama/Llama-4-Scout-17B-16E-Instruct`. List models your token can use:
=======
| Variable   | Required | Default                         | Notes |
|------------|----------|---------------------------------|-------|
| `HF_TOKEN` | yes      | —                               | Hugging Face access token. |
| `HF_MODEL` | no       | `Qwen/Qwen3-VL-8B-Instruct`     | Any **image-capable** model served by a provider you've enabled. |

Other vision models that work well (use a larger one for messy handwritten photos):
`Qwen/Qwen3-VL-30B-A3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`,
`google/gemma-3-27b-it`, `meta-llama/Llama-4-Scout-17B-16E-Instruct`.

To see exactly which models your token can use, list them with:
>>>>>>> a062e5639debf1cb57716998e5203e220042c921
`curl -H "Authorization: Bearer $HF_TOKEN" https://router.huggingface.co/v1/models`

---

<<<<<<< HEAD
## How accuracy is maximized

- **Deterministic OMR.** Present/absent is decided by measuring each circle's
  interior fill — not by an LLM — so it can't hallucinate and accuracy doesn't
  degrade as the sheet grows. On clean printed 20×30 / 30-day sheets this is
  effectively 100%.
- **Resolution-robust grid detection.** Small uploads are upscaled and thin
  1-pixel rule lines are bridged before line extraction, so dense grids don't
  collapse; line lengths are judged relative to the table, so merged header
  banners and multi-row headers don't drop day columns.
- **Per-sheet adaptive threshold.** A k-means split over every cell's fill ratio
  separates filled vs empty *for that sheet* (printed, pen, pencil), and
  borderline cells are marked **unclear** rather than guessed.
- **Confidence gating + tiled fallback.** The reader auto-returns only when the
  grid is clean and the marks separate well; otherwise it falls back to the
  vision model **in small tiles** so even hard photos never trigger whole-grid
  hallucination.

### Tips for best results
- Photograph/scan the sheet flat, well-lit, filling the frame; one sheet per image.
- Printed/ruled grids read best; purely hand-drawn wavy grids are harder and may
  fall back to the AI reader.
- If roll numbers come back as `1, 2, 3…`, set `HF_TOKEN` to enable OCR.

---

## Tests

A synthetic accuracy harness lives in [`tests/`](tests) (see
[`tests/README.md`](tests/README.md)):

```bash
python tests/generate_sheets.py    # creates tests/_synth/ (gitignored)
python tests/eval_omr.py           # measures cell-level accuracy, offline
```

It renders sheets up to 20×30 across six capture conditions (clean, rotated,
perspective, blur+noise, low-res JPEG, combo) and verifies the deterministic
path is ~100% on the cases it auto-returns, abstaining (→ fallback) on the rest.
=======
## Try it without your own sheet

Two example images ship in [`static/samples/`](static/samples) and appear as
**"Try an example"** thumbnails on the upload screen — click one, then
**Analyze attendance**. Great for testing the deployed link.

| Sample | What it tests |
|--------|---------------|
| `printed-sheet.png` | Clean printed register (8 students × 3 days) — the easy case. |
| `handwritten-photo.png` | Tilted phone photo of a hand-drawn sheet — tests preprocessing + robustness. |

Answer key for `printed-sheet.png` (P = present, A = absent):

| Student | Mon 23 | Tue 24 | Wed 25 |
|---------|:--:|:--:|:--:|
| Aarav Sharma | P | P | A |
| Diya Patel | A | P | P |
| Kabir Singh | P | A | P |
| Ananya Rao | P | P | A |
| Vivaan Mehta | A | A | P |
| Ishita Nair | P | A | A |
| Rohan Das | A | P | P |
| Sara Khan | P | P | A |

The default model reads this sheet with 100% accuracy in both normal and
high-contrast mode.

---

## How accuracy is maximized

- **Preprocessing** removes camera noise and lifts faint pencil marks before the
  model ever sees the image. Use the **High-contrast (binarize)** toggle for very
  pale or low-quality scans.
- The model is prompted to return **strict JSON** and to mark genuinely
  ambiguous cells as `"unclear"` (shown in amber) rather than guessing.

### Tips for best results
- Photograph the sheet flat, well-lit, filling the frame.
- One sheet per image.
- If names come back garbled, retake the photo closer / sharper.
>>>>>>> a062e5639debf1cb57716998e5203e220042c921

---

## API reference

<<<<<<< HEAD
| Method | Path           | Body (multipart/form-data)            | Returns |
|--------|----------------|---------------------------------------|---------|
| `GET`  | `/`            | —                                     | UI |
| `POST` | `/api/preview` | `image`, `binarize` (0/1)             | `{ processed_image }` (data URI) |
| `POST` | `/api/analyze` | `image`, `students` (opt), `days` (opt) | `{ processed_image, result, model }` |
| `GET`  | `/healthz`     | —                                     | `{ status, token_configured }` |

`students` / `days` are optional integer hints (no UI field): when provided, a
detected grid whose size doesn't match is treated as low-confidence and routed
to the fallback instead of being returned.

`result` shape (keyed by **roll number**, never by name):

```json
{
  "source": "omr",
  "roll_source": "ocr",
  "confident": true,
  "columns": ["Col 1", "Col 2"],
  "dates": ["Col 1", "Col 2"],
  "students": [
    { "roll": "23", "name": null, "roll_confidence": "high",
      "attendance": { "Col 1": "present", "Col 2": "absent" } }
  ],
  "confidence": { "grid_ok": true, "n_rows": 20, "n_cols": 30,
                  "grid_confidence": 0.98, "separation_margin": 0.42,
                  "ambiguous_cells": 1 },
  "warnings": [],
  "review": [ { "roll": "23", "date": "Col 2" } ]
}
```

- `source` — `"omr"` (deterministic OpenCV), `"vlm"` (tiled AI fallback) or `"hybrid"`.
- `roll_source` — `"ocr"` (read from the sheet) or `"index"` (row numbers).
- `name` is always `null` (names are never extracted).
- `review` lists `unclear` cells for a quick human check.
- `dates` is an alias of `columns`; date-header labels default to `Col 1…N`.
=======
| Method | Path           | Body (multipart/form-data)        | Returns |
|--------|----------------|-----------------------------------|---------|
| `GET`  | `/`            | —                                 | UI |
| `POST` | `/api/preview` | `image`, `binarize` (0/1)         | `{ processed_image }` (data URI) |
| `POST` | `/api/analyze` | `image`, `binarize` (0/1)         | `{ processed_image, result, model }` |
| `GET`  | `/healthz`     | —                                 | `{ status, token_configured }` |

`result` shape:

```json
{
  "dates": ["2026-06-25"],
  "students": [
    { "name": "Asha R", "attendance": { "2026-06-25": "present" } },
    { "name": "Vikram S", "attendance": { "2026-06-25": "absent" } }
  ]
}
```

