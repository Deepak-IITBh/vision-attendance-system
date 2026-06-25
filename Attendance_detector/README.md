# 📋 Attendance Detector

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

---

## Project structure

```
Attendance_detector/
├── app.py              # Flask app + API routes
├── preprocess.py       # OpenCV image cleanup
├── hf_client.py        # Hugging Face Inference Providers client
├── requirements.txt
├── render.yaml         # Render blueprint
├── Procfile
├── runtime.txt         # pins Python 3.12 for stable wheels
├── .env.example
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

---

## 1. Get a Hugging Face token (free)

1. Sign in / create an account at <https://huggingface.co>.
2. Go to **Settings → Access Tokens** → <https://huggingface.co/settings/tokens>.
3. Create a token with **Read** access and copy it (`hf_...`).

The default model is `Qwen/Qwen3-VL-8B-Instruct`, served through HF
**Inference Providers**. A free account includes a monthly inference credit. A
model only works if a provider you've enabled serves it — check/enable providers
at <https://huggingface.co/settings/inference-providers>. If you hit a "not
supported by any provider you have enabled" error, switch `HF_MODEL` (see below).

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

cp .env.example .env          # then edit .env and paste your HF_TOKEN
python app.py
```

Open <http://localhost:5000>, upload a sheet, click **Preview filter** to see the
cleaned image, then **Analyze attendance**.

Health check: <http://localhost:5000/healthz> (shows whether the token is set).

---

## 3. Deploy to Render

### Option A — Blueprint (recommended)
1. Push this folder to a GitHub repo.
2. In Render: **New → Blueprint**, pick the repo. Render reads `render.yaml`.
3. When prompted, set the **`HF_TOKEN`** environment variable to your token
   (it's marked `sync: false`, so it is never stored in the repo).
4. Deploy. The public URL serves the same UI.

### Option B — Manual Web Service
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT`
- **Environment variables:** `HF_TOKEN` (required), `HF_MODEL` (optional),
  `PYTHON_VERSION=3.12.7`

> The long gunicorn `--timeout` matters: vision-model calls take ~10–30s and the
> default 30s timeout would otherwise kill the request.
>
> On Render's **free** plan the service sleeps after inactivity, so the first
> request after idle may take a while to wake up.

---

## Configuration

| Variable   | Required | Default                         | Notes |
|------------|----------|---------------------------------|-------|
| `HF_TOKEN` | yes      | —                               | Hugging Face access token. |
| `HF_MODEL` | no       | `Qwen/Qwen3-VL-8B-Instruct`     | Any **image-capable** model served by a provider you've enabled. |

Other vision models that work well (use a larger one for messy handwritten photos):
`Qwen/Qwen3-VL-30B-A3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`,
`google/gemma-3-27b-it`, `meta-llama/Llama-4-Scout-17B-16E-Instruct`.

To see exactly which models your token can use, list them with:
`curl -H "Authorization: Bearer $HF_TOKEN" https://router.huggingface.co/v1/models`

---

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

---

## API reference

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
