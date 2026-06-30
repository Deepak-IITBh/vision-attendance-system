# 📋 Attendance Detector

Upload a photo or scan of a class **attendance sheet** and get back a clean
present/absent table keyed by **roll number**. The app uses a deterministic
OpenCV-based OMR engine for attendance marks and an optional Hugging Face
fallback for hard-to-read roll numbers or difficult photos.

> Flask backend · OpenCV OMR engine · HTML/CSS/JS frontend · deploys to Render.

---

## How it works

1. **Preprocess image** — resize, denoise, enhance contrast, deskew, and
   optionally binarize so circles become easier to read.
2. **Detect the table grid** — identify rows, columns, and headers using rule
   lines and table structure.
3. **Read attendance marks** — classify each date cell by circle fill: a
   **filled/shaded/scribbled circle = present**, an **open circle = absent**.
4. **Optional fallback OCR** — roll numbers are read with a Hugging Face vision
   model only when needed.
5. **Return structured JSON** — the result includes roll numbers and daily
   attendance state.

---

## Project structure

```
Attendance_detector/
├── app.py              # Flask app + API routes
├── omr.py              # OpenCV OMR engine (rectify, grid, roles, mark reading, fallback)
├── preprocess.py       # OpenCV image cleanup and helper utilities
├── hf_client.py        # Hugging Face inference client for OCR/fallback
├── requirements.txt
├── render.yaml         # Render blueprint
├── Procfile
├── runtime.txt         # Python runtime pin
├── .env.example
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

---

## Requirements

- Python 3.12 or 3.13
- pip
- Optional: Hugging Face token for OCR and fallback

---

## Run locally

```bash
cd Attendance_detector
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
copy .env.example .env
# edit .env and add HF_TOKEN if you want OCR and fallback support
python app.py
```

Open:

- `http://localhost:5000`
- `http://localhost:5000/healthz`

---

## Configuration

| Variable   | Required | Default                     | Notes |
|------------|----------|-----------------------------|-------|
| `HF_TOKEN` | no       | —                           | Hugging Face token for OCR and fallback. |
| `HF_MODEL` | no       | `Qwen/Qwen3-VL-8B-Instruct` | Optional image-capable model. |

If you set `HF_TOKEN`, make sure the selected model is supported by one of your
enabled Hugging Face inference providers.

To list available models for your token:

```bash
curl -H "Authorization: Bearer $HF_TOKEN" https://router.huggingface.co/v1/models
```

---

## Deploy to Render

### Option A — Blueprint (recommended)
1. Push the repo to GitHub.
2. In Render, choose **New → Blueprint**, then select the repo.
3. Set `HF_TOKEN` as an environment variable when prompted.
4. Deploy.

### Option B — Manual Web Service
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT`
- Environment variables: `HF_TOKEN` (optional), `HF_MODEL` (optional), `PYTHON_VERSION=3.12.7`

> Use `--timeout 120` because long OCR or fallback requests can take more time
> than the default gunicorn timeout.

---

## Notes

- Attendance marks are read with deterministic OpenCV OMR.
- The vision model is used only for roll-number OCR or fallback on hard images.
- If roll numbers appear as `1, 2, 3…`, the app did not receive a valid
  `HF_TOKEN` or could not OCR the numbers reliably.
- Keep uploads under 16 MB.
