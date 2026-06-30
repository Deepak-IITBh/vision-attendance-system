"""Image preprocessing for attendance sheets.

The goal is to make hand-drawn / printed circles and handwritten names as clear
as possible before sending the image to the vision model. We deliberately keep a
grayscale (not harshly binarized) output by default, because vision-language
models read natural-looking images better than aggressively thresholded ones.
A `binarize` option is exposed for very low-contrast scans.
"""

import base64

import cv2
import numpy as np

# Cap the longest side so payloads stay small and processing stays fast.
MAX_DIM = 1600

# Reject absurdly large rasters early: a 16 MB upload can still decode to a
# multi-hundred-megapixel image (PNG/WEBP compress heavily), which would OOM a
# small dyno. 40 MP comfortably covers any real phone photo / scan.
MAX_PIXELS = 40_000_000


def _read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image. Please upload a valid JPG, PNG or WEBP file.")
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError("Image resolution is too high. Please upload a smaller photo (under ~40 megapixels).")
    return img


def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > MAX_DIM:
        scale = MAX_DIM / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _estimate_skew(gray: np.ndarray):
    """Estimate a small skew angle from the dominant text/lines. Returns None if
    the angle is negligible or implausibly large (likely a misdetection)."""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 100:
        return None
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 30:
        return None
    return angle


def _deskew(gray: np.ndarray) -> np.ndarray:
    angle = _estimate_skew(gray)
    if angle is None:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def enhance(file_bytes: bytes, binarize: bool = False) -> np.ndarray:
    """Return a cleaned single-channel image ready for the model / preview."""
    img = _resize(_read_image(file_bytes))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Reduce camera noise.
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 2. Local contrast boost so faint pencil circles become visible.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3. Straighten a tilted phone photo.
    gray = _deskew(gray)

    # 4. Unsharp mask to crisp up edges of circles and text.
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    if binarize:
        # IMPORTANT: do NOT use adaptive thresholding here. Adaptive threshold
        # compares each pixel to its local neighbourhood, which hollows out any
        # solid region larger than the window — a FILLED circle becomes a ring,
        # destroying the present/absent signal. Instead we flatten uneven
        # lighting by dividing out a blurred background estimate, then apply a
        # global Otsu threshold, which keeps filled bubbles solid.
        bg = cv2.GaussianBlur(sharp, (0, 0), sigmaX=25)
        norm = cv2.divide(sharp, bg, scale=255)
        return cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return sharp


def to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode processed image.")
    return buf.tobytes()


def to_data_uri(img: np.ndarray) -> str:
    b64 = base64.b64encode(to_png_bytes(img)).decode("ascii")
    return "data:image/png;base64," + b64
