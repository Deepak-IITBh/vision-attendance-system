"""Flask backend for the Attendance Detector.

Endpoints:
  GET  /            -> the single-page UI
  POST /api/preview -> returns the preprocessed image (so the user can see the
                       filter before spending an inference call)
  POST /api/analyze -> preprocess + send to the vision model, returns the
                       attendance table as JSON
  GET  /healthz     -> health check for Render
"""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

import hf_client
import omr
import preprocess

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap

ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


def _get_upload():
    if "image" not in request.files:
        return None, ("No image uploaded. Please choose a file.", 400)
    f = request.files["image"]
    if not f or f.filename == "":
        return None, ("No image selected.", 400)
    if f.mimetype not in ALLOWED:
        return None, ("Unsupported file type. Please upload a JPEG, PNG or WEBP image.", 400)
    data = f.read()
    if not data:
        return None, ("Uploaded file is empty.", 400)
    return data, None


def _wants_binarize() -> bool:
    return str(request.form.get("binarize", "")).lower() in {"1", "true", "on", "yes"}


def _int_form(name: str):
    """Optional positive integer from the form (expected #students / #days)."""
    raw = (request.form.get(name) or "").strip()
    if not raw:
        return None
    try:
        val = int(raw)
        return val if 0 < val <= 1000 else None
    except ValueError:
        return None


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_exc):
    return jsonify(error="Image is too large. Please upload a file under 16 MB."), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preview", methods=["POST"])
def preview():
    data, err = _get_upload()
    if err:
        return jsonify(error=err[0]), err[1]
    try:
        processed = preprocess.enhance(data, binarize=_wants_binarize())
        return jsonify(processed_image=preprocess.to_data_uri(processed))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("preview failed")
        return jsonify(error=f"Preprocessing failed: {exc}"), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data, err = _get_upload()
    if err:
        return jsonify(error=err[0]), err[1]

    try:
        result = omr.analyze(
            data,
            expected_rows=_int_form("students"),
            expected_cols=_int_form("days"),
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("analyze failed")
        return jsonify(error=f"Analysis failed: {exc}"), 500

    if not result.get("ok", True):
        return jsonify(error=result.get("error", "Could not read the sheet."),
                       result=result), 422

    model = "OpenCV OMR" if result.get("source") == "omr" else \
        os.environ.get("HF_MODEL", hf_client.DEFAULT_MODEL)
    return jsonify(
        processed_image=result.get("processed_image"),
        result=result,
        model=model,
    )


@app.route("/healthz")
def healthz():
    return jsonify(status="ok", token_configured=bool(os.environ.get("HF_TOKEN")))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
