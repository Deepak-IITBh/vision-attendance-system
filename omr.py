"""Deterministic OpenCV attendance reader (OMR).

This is the accuracy core. Instead of asking a vision-language model to read a
whole 600-cell grid at once (which makes it hallucinate), we:

  1. Rectify the sheet  - perspective-warp a tilted phone photo to a clean
     fronto-parallel rectangle, or just deskew an axis-aligned scan.
  2. Detect the grid    - find horizontal/vertical rule lines and rebuild the
     matrix of cell rectangles.
  3. Assign roles       - which column holds the roll number vs the date marks,
     and which row is the header.
  4. Classify marks     - for every date cell, measure the *interior* fill of
     the circle (a filled circle = present, an open ring = absent) with a
     per-sheet adaptive threshold and an "unclear" abstain band.
  5. Score confidence   - only trust the deterministic result when the grid is
     clean and the present/absent fill ratios separate cleanly.

The vision model is still used, but only for small, reliable sub-tasks: reading
the roll-number column from a narrow cropped strip, and (in the fallback path)
reading the sheet in small chunks. See ``hf_client`` and ``orchestrator``-style
flow in :func:`analyze`.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

import hf_client
import preprocess

# Run the CV stages at a controlled resolution. Classical detection wants crisp
# cells AND rule lines that are a few pixels thick: a 1px anti-aliased divider in
# a low-res upload fragments under morphology and dense grids collapse. So we
# work in a [MIN, MAX] band - downscale big phone photos, but also UPSCALE small
# uploads so thin lines thicken enough to survive.
OMR_MAX_DIM = 2400
OMR_MIN_DIM = 2200

GRID_CONF_MIN = 0.45        # below this we don't trust the detected grid
SEPARATION_MIN = 0.20       # min gap between filled/empty fill-ratio clusters
AMBIGUOUS_FRAC_MAX = 0.03   # fraction of cells allowed in the "unclear" band
MARK_COL_FRAC = 0.5         # >=50% of a column's DATA cells hold a circle -> mark column
ID_COL_FRAC = 0.35          # <=35% -> identity (roll/name) column


# --------------------------------------------------------------------------- #
# 1. Rectification (perspective warp / deskew)                                #
# --------------------------------------------------------------------------- #
def _read_resize(file_bytes: bytes) -> np.ndarray:
    img = preprocess._read_image(file_bytes)
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > OMR_MAX_DIM:
        s = OMR_MAX_DIM / longest
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    elif longest < OMR_MIN_DIM:
        s = OMR_MIN_DIM / longest
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return img


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], np.float32)


def _angles_ok(q: np.ndarray, lo: float = 55, hi: float = 125) -> bool:
    def ang(a, b, c):
        v1, v2 = a - b, c - b
        cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    vals = [ang(q[(i - 1) % 4], q[i], q[(i + 1) % 4]) for i in range(4)]
    return all(lo <= v <= hi for v in vals)


def _find_sheet_quad(gray: np.ndarray):
    """Largest trustworthy 4-corner quadrilateral, in full-res gray coords."""
    h, w = gray.shape
    scale = 700.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    frame = small.shape[0] * small.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if not (0.25 * frame <= area <= 0.985 * frame):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(approx))
        if hull_area <= 0 or area / hull_area < 0.9:
            continue
        q = _order_corners(approx)
        sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
        if min(sides) < 0.2 * max(sides) or not _angles_ok(q):
            continue
        return q / scale
    return None


def _warp(img: np.ndarray, q: np.ndarray):
    tl, tr, br, bl = q
    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if maxW < 50 or maxH < 50:
        return None
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return M, (maxW, maxH)


def _line_skew_angle(gray: np.ndarray) -> float:
    """Rotation of the grid, in degrees, from its own rule lines. We Hough the
    raw edges (NOT a morphological horizontal mask, which would erase any line
    tilted more than ~1deg) and fold every long segment's angle into [-45,45):
    both the horizontal and vertical rules of a grid rotated by theta fold to
    theta, so their median is a robust skew estimate. This is exactly what the
    projection-based detector needs zeroed - a 1deg tilt smears a line across
    dozens of rows and it drops below threshold."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    minlen = max(40, min(h, w) // 6)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=120,
                            minLineLength=minlen, maxLineGap=10)
    if lines is None:
        return 0.0
    angs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        folded = ((a + 45) % 90) - 45        # grid lines (h & v) both -> theta
        if abs(folded) < 20:
            angs.append(folded)
    return float(np.median(angs)) if angs else 0.0


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _is_perspective(q: np.ndarray) -> bool:
    """True only for genuine perspective (converging edges). Pure rotation has
    equal opposite sides and is corrected far more reliably by line-deskew than
    by a 4-point warp built from approximate corners."""
    tl, tr, br, bl = q
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    return (abs(top - bottom) / max(top, bottom, 1) > 0.035 or
            abs(left - right) / max(left, right, 1) > 0.035)


def rectify(file_bytes: bytes):
    """Return (gray, bgr, rectified_bool). Perspective-warp only a genuinely
    skewed photo; then zero any rotation using the rule lines themselves
    (projection-based grid detection needs the lines truly axis-aligned)."""
    bgr = _read_resize(file_bytes)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rectified = False

    q = _find_sheet_quad(gray)
    if q is not None and _is_perspective(q):
        warp = _warp(gray, q)
        if warp is not None:
            M, size = warp
            gray = cv2.warpPerspective(gray, M, size, flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
            bgr = cv2.warpPerspective(bgr, M, size, flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
            rectified = True

    # Primary / residual rotation correction from the grid lines themselves.
    angle = _line_skew_angle(gray)
    if abs(angle) > 0.1:
        gray = _rotate(gray, angle)
        bgr = _rotate(bgr, angle)
        rectified = True
    return gray, bgr, rectified


# --------------------------------------------------------------------------- #
# 2. Grid detection                                                           #
# --------------------------------------------------------------------------- #
def _enhance(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _line_centers(mask: np.ndarray, axis: int, min_span_px: float, band=None):
    """Centre coordinate of every rule line in a morphological line mask.

    A line is any connected component spanning at least ``min_span_px`` along its
    axis (length-based, not magnitude-based, so a faint/slightly-tilted line
    still registers instead of dropping below a projection threshold).

    For vertical lines, ``band=(top, bottom)`` measures the span only INSIDE the
    table's vertical extent. This is essential: when the header has a merged cell
    (e.g. a "JUNE (1-30)" banner spanning all day columns), the day-column
    dividers are SHORTER than the full-height Roll/Name dividers, and an
    image-relative threshold would drop every day line and collapse 30 columns
    into one. A title/legend stroke outside the band contributes ~0 span.

    axis=0 -> horizontal lines (return y centres); axis=1 -> vertical (x)."""
    num, _labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    coords = []
    for i in range(1, num):
        y, wc, hc = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if axis == 0:
            if wc >= min_span_px:
                coords.append(int(round(float(cent[i][1]))))
        else:
            span = hc if band is None else (min(y + hc, band[1]) - max(y, band[0]))
            if span >= min_span_px:
                coords.append(int(round(float(cent[i][0]))))
    return sorted(coords)


def _merge_close(coords, min_frac=0.3):
    """Collapse lines closer than ``min_frac`` of the median spacing into one
    (kills paper-edge slivers and doubled lines from dilation)."""
    coords = sorted(coords)
    if len(coords) < 3:
        return coords
    med = float(np.median(np.diff(coords)))
    if med <= 0:
        return coords
    out = [coords[0]]
    for c in coords[1:]:
        if c - out[-1] < min_frac * med:
            out[-1] = int(round((out[-1] + c) / 2.0))
        else:
            out.append(c)
    return out


def _trim_edge_bands(coords):
    """Drop the outermost line when it forms a tiny edge band - a desk/paper
    margin left over after a perspective warp. (An identity column is *wider*,
    never narrower, and a missing-line band is wider, so a sub-median edge band
    is reliably a sliver.)"""
    coords = sorted(coords)
    if len(coords) < 4:
        return coords
    med = float(np.median(np.diff(coords)))
    if med <= 0:
        return coords
    if coords[1] - coords[0] < 0.4 * med:
        coords = coords[1:]
    if len(coords) >= 4 and coords[-1] - coords[-2] < 0.4 * med:
        coords = coords[:-1]
    return coords


def _fill_regular(coords):
    """Insert lines into gaps that are clean integer multiples of the median
    spacing (recovers a faint/missing interior rule line)."""
    if len(coords) < 3:
        return coords
    coords = sorted(coords)
    gaps = np.diff(coords).astype(float)
    med = float(np.median(gaps))
    if med <= 0:
        return coords
    out = [coords[0]]
    for i in range(1, len(coords)):
        g = coords[i] - coords[i - 1]
        n = int(round(g / med))
        # Only fill gaps that are a clean integer multiple of the spacing (a
        # missing rule line). A wide identity column (Roll No / Name) is NOT a
        # clean multiple, so the tight residual keeps it from being split.
        if n >= 2 and abs(g - n * med) < 0.12 * med:
            for k in range(1, n):
                out.append(int(round(coords[i - 1] + k * med)))
        out.append(coords[i])
    return out


def _reg(coords, drop_max: bool = False) -> float:
    g = np.diff(sorted(coords)).astype(float)
    if len(g) == 0:
        return 0.0
    if drop_max and len(g) > 2:
        g = np.sort(g)[:-1]   # ignore the wide identity column
    med = float(np.median(g))
    if med <= 0:
        return 0.0
    return max(0.0, 1.0 - float(np.median(np.abs(g - med))) / med)


def detect_grid(gray: np.ndarray, expected_rows=None, expected_cols=None):
    """Detect rule lines and return the cell matrix + a confidence score."""
    h, w = gray.shape
    bs = max(15, int(round(min(h, w) / 30)))
    if bs % 2 == 0:
        bs += 1
    binar = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, bs, 10)
    # NOTE: no medianBlur here - a 3x3 median erases 1px-wide rule lines
    # (outvoted by surrounding white), which collapses dense hairline grids. The
    # directional OPEN below already discards specks shorter than the line kernel.

    # Horizontal rules first - they span most of the table width and pin the
    # table's vertical extent (top/bottom), which the vertical pass needs.
    hk = max(20, w // 40)
    # First bridge the small along-line gaps a thin/anti-aliased rule line breaks
    # into after thresholding (otherwise the OPEN erases the broken line before
    # the length filter ever sees it). The close kernel is tiny so it can't weld
    # neighbouring text vertically; and even if it joined a few letters of a word
    # horizontally, the 0.45*width span filter rejects it as a "line".
    horiz = cv2.morphologyEx(binar, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 350), 1)))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, hk // 4), 1)))
    ys = _merge_close(_line_centers(horiz, axis=0, min_span_px=0.45 * w))
    if len(ys) < 3:
        return None
    gtop, gbot = ys[0], ys[-1]
    gh = max(1, gbot - gtop)

    # Vertical rules sized AND thresholded relative to the table height (not the
    # image height) and measured only inside [gtop, gbot] - so short merged-
    # header day dividers survive and title/legend strokes don't intrude.
    vk = max(12, gh // 25)
    # Bridge along-line AA gaps first (1px-wide kernel -> cannot weld separate
    # text columns horizontally), then keep only long vertical runs.
    vert = cv2.morphologyEx(binar, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 350))))
    vert = cv2.morphologyEx(vert, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, vk // 4))))
    xs = _merge_close(_line_centers(vert, axis=1, min_span_px=0.4 * gh, band=(gtop, gbot)))
    if len(xs) < 3:
        return None

    ys = _fill_regular(_trim_edge_bands(ys))
    xs = _fill_regular(_trim_edge_bands(xs))

    if len(xs) < 3 or len(ys) < 3:
        return None

    n_rows = len(ys) - 1
    n_cols = len(xs) - 1
    cells = [[(xs[c], ys[r], xs[c + 1], ys[r + 1]) for c in range(n_cols)]
             for r in range(n_rows)]

    regularity = 0.5 * _reg(ys) + 0.5 * _reg(xs, drop_max=True)
    count_ok = 1.0 if (n_rows >= 2 and n_cols >= 2) else 0.0
    confidence = count_ok * regularity

    return {"cells": cells, "xs": xs, "ys": ys, "n_rows": n_rows,
            "n_cols": n_cols, "confidence": confidence}


# --------------------------------------------------------------------------- #
# 3. Cell binarization + circle / fill measurement                            #
# --------------------------------------------------------------------------- #
def _binarize_cell(roi: np.ndarray) -> np.ndarray:
    """Ink = 255. Flatten lighting by dividing out a blurred background, then
    Otsu. (Adaptive threshold is deliberately avoided - it hollows solid fills,
    turning a filled circle into a ring; see preprocess.py.)"""
    sigma = max(3.0, max(roi.shape) / 4.0)
    bg = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma)
    norm = cv2.divide(roi, bg, scale=255)
    _, binv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binv


def _roi(gray, box, margin_frac):
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    if cw < 6 or ch < 6:
        return None
    mx, my = int(margin_frac * cw), int(margin_frac * ch)
    sub = gray[y0 + my:y1 - my, x0 + mx:x1 - mx]
    return sub if sub.size else None


def _has_circle(gray, box) -> bool:
    """Does the cell contain one centered, roughly circular object? (Used only
    to tell mark columns from text columns - fill state is irrelevant here.)"""
    roi = _roi(gray, box, 0.12)
    if roi is None:
        return False
    binv = _binarize_cell(roi)
    rh, rw = roi.shape
    area_cell = rh * rw
    cnts, _ = cv2.findContours(binv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.08 * area_cell or a > 0.90 * area_cell:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4 * np.pi * a / (peri * peri)
        if circ < 0.55:
            continue
        # Must be a real bubble, not a round text glyph ("0"/"D"/"8" in a date
        # header blur into circle-ish blobs otherwise and the header row gets
        # mistaken for a data row).
        (_cx, _cy), er = cv2.minEnclosingCircle(c)
        if er < 0.20 * min(rw, rh):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        if 0.2 * rw < cx < 0.8 * rw and 0.2 * rh < cy < 0.8 * rh:
            return True
    return False


def _fill_ratio(roi: np.ndarray):
    """Return (interior_fill_ratio, total_ink_ratio, n_circles). The interior
    fill of the circle's inner disc separates a solid/scribbled circle (high)
    from an open ring (~0, its ink sits on the boundary, outside the inner
    disc). ``n_circles`` counts distinct circle-like blobs - >1 means two
    columns merged into one cell (grid under-segmented)."""
    binv = _binarize_cell(roi)
    rh, rw = roi.shape
    total = float((binv > 0).mean())
    if total < 0.02:
        return 0.0, total, 0          # blank cell -> absent

    cnts, _ = cv2.findContours(binv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_cell = rh * rw
    n_circles = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.04 * area_cell:
            continue
        peri = cv2.arcLength(c, True)
        if peri > 0 and 4 * np.pi * a / (peri * peri) > 0.5:
            n_circles += 1

    cx, cy, rad = rw / 2.0, rh / 2.0, 0.5 * min(rw, rh)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (ecx, ecy), er = cv2.minEnclosingCircle(c)
        if er > 0.2 * min(rw, rh):
            cx, cy, rad = ecx, ecy, er

    r_in = max(1, int(round(0.60 * rad)))
    mask = np.zeros(binv.shape, np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), r_in, 255, -1)
    disc = cv2.countNonZero(mask)
    if disc == 0:
        return 0.0, total, n_circles
    inside = cv2.countNonZero(cv2.bitwise_and(binv, mask))
    return inside / disc, total, n_circles


# --------------------------------------------------------------------------- #
# 4. Role assignment                                                          #
# --------------------------------------------------------------------------- #
def assign_roles(gray, cells):
    """Classify columns as identity vs date-mark and keep only genuine data rows.

    Order matters: find DATA ROWS first (rows packed with circles), then judge
    columns over data rows only - otherwise header rows (which may be several,
    e.g. a "JUNE (1-30)" banner + a day-number row) dilute every column's
    circle-fraction and legitimate day columns fall below the threshold. The
    date region is then taken as the CONTIGUOUS span between the first and last
    circle-bearing columns, so a day column with a few missed/ambiguous circles
    is still kept instead of splitting the table."""
    n_rows, n_cols = len(cells), len(cells[0])
    has = [[_has_circle(gray, cells[r][c]) for c in range(n_cols)] for r in range(n_rows)]

    row_circ = [sum(has[r]) for r in range(n_rows)]
    max_rc = max(row_circ) if row_circ else 0
    if max_rc == 0:
        return None
    # A data row holds a circle in (almost) every mark column; a header/title
    # band only has circles where a label happens to contain a round glyph
    # ("0"/"D"/"8" in date headers), which is far fewer. 0.6*max cleanly splits
    # the two without dropping a data row that merely lost a few circles.
    data_rows = [r for r in range(n_rows) if row_circ[r] >= 0.6 * max_rc]
    if not data_rows:
        return None

    def col_frac(c):
        return float(np.mean([has[r][c] for r in data_rows]))

    strong = [c for c in range(n_cols) if col_frac(c) >= MARK_COL_FRAC]
    if not strong:
        return None
    lo, hi = min(strong), max(strong)
    mark_cols = list(range(lo, hi + 1))           # contiguous date region

    identity = [c for c in range(lo) if col_frac(c) <= ID_COL_FRAC]
    roll_col = identity[0] if identity else None
    return {"header_row": None, "data_rows": data_rows,
            "mark_cols": mark_cols, "roll_col": roll_col}


# --------------------------------------------------------------------------- #
# 5. Mark classification with a per-sheet adaptive threshold                  #
# --------------------------------------------------------------------------- #
def classify_marks(gray, cells, data_rows, mark_cols):
    feats, ratios = {}, []
    multi = 0
    for r in data_rows:
        for c in mark_cols:
            roi = _roi(gray, cells[r][c], 0.18)
            if roi is None:
                feats[(r, c)] = None
                continue
            fr, _total, n_circ = _fill_ratio(roi)
            feats[(r, c)] = fr
            ratios.append(fr)
            if n_circ >= 2:
                multi += 1

    arr = np.array(ratios, np.float32)
    if len(arr) >= 4 and float(arr.std()) > 1e-3:
        Z = arr.reshape(-1, 1)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, _labels, centers = cv2.kmeans(Z, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        c_lo, c_hi = sorted(float(x) for x in centers.ravel())
    else:
        c_lo, c_hi = 0.0, 1.0
    spread = c_hi - c_lo
    t = (c_lo + c_hi) / 2.0

    states, ambiguous = {}, 0
    if spread < SEPARATION_MIN:
        t = 0.35                              # unimodal sheet: absolute cutoff
        for k, fr in feats.items():
            states[k] = "unclear" if fr is None else ("present" if fr > t else "absent")
    else:
        band = max(0.05, 0.15 * spread)
        for k, fr in feats.items():
            if fr is None:
                states[k] = "unclear"
                ambiguous += 1
            elif abs(fr - t) < band:
                states[k] = "unclear"
                ambiguous += 1
            else:
                states[k] = "present" if fr > t else "absent"

    total = max(1, len(arr))
    stats = {"spread": spread, "threshold": t, "ambiguous": ambiguous,
             "ambiguous_frac": ambiguous / total, "n_cells": len(arr),
             "multi_circle_frac": multi / total}
    return states, stats


# --------------------------------------------------------------------------- #
# 6. Roll numbers (tiny VLM strip OCR, with index fallback)                   #
# --------------------------------------------------------------------------- #
def _strip_data_uri(bgr, cells, col, data_rows, upscale=2.0):
    x0 = cells[data_rows[0]][col][0]
    x1 = cells[data_rows[0]][col][2]
    y0 = cells[data_rows[0]][col][1]
    y1 = cells[data_rows[-1]][col][3]
    pad = int(0.02 * (y1 - y0))
    y0 = max(0, y0 - pad)
    y1 = min(bgr.shape[0], y1 + pad)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    if upscale and upscale != 1.0:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        if max(crop.shape[:2]) > 1600:
            s = 1600 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return preprocess.to_data_uri(crop)


def read_rolls(bgr, cells, roles, allow_vlm: bool):
    n = len(roles["data_rows"])
    if roles["roll_col"] is None or not allow_vlm:
        return [str(i + 1) for i in range(n)], "index"
    uri = _strip_data_uri(bgr, cells, roles["roll_col"], roles["data_rows"])
    if uri is None:
        return [str(i + 1) for i in range(n)], "index"
    try:
        vals = hf_client.read_roll_strip(uri, n)
    except Exception:
        vals = None
    if not vals:
        return [str(i + 1) for i in range(n)], "index"
    rolls = [str(vals[i]).strip() if i < len(vals) and str(vals[i]).strip() else str(i + 1)
             for i in range(n)]
    return _dedupe(rolls), "ocr"


def _dedupe(rolls):
    seen, out = {}, []
    for r in rolls:
        if r in seen:
            seen[r] += 1
            out.append(f"{r}#{seen[r]}")
        else:
            seen[r] = 1
            out.append(r)
    return out


# --------------------------------------------------------------------------- #
# 7. Assemble + orchestrate                                                   #
# --------------------------------------------------------------------------- #
def _columns(mark_cols):
    return [f"Col {i + 1}" for i in range(len(mark_cols))]


def _assemble(rolls, roll_source, columns, states, data_rows, mark_cols,
              source, grid, stats, warnings):
    matrix, students, review = [], [], []
    for i, r in enumerate(data_rows):
        row_states = []
        attendance = {}
        for j, c in enumerate(mark_cols):
            st = states.get((r, c), "unclear")
            row_states.append(st)
            attendance[columns[j]] = st
            if st == "unclear":
                review.append({"roll": rolls[i], "date": columns[j]})
        matrix.append(row_states)
        students.append({"roll": rolls[i], "name": None,
                         "roll_confidence": "high" if roll_source == "ocr" else "low",
                         "attendance": attendance})
    return {
        "source": source,
        "has_roll_column": roll_source == "ocr",
        "roll_source": roll_source,
        "columns": columns,
        "dates": columns,                      # alias for older frontend
        "students": students,
        "confidence": {
            "grid_ok": grid is not None,
            "n_rows": len(data_rows),
            "n_cols": len(mark_cols),
            "grid_confidence": round(float(grid["confidence"]), 3) if grid else 0.0,
            "separation_margin": round(float(stats.get("spread", 0.0)), 3),
            "ambiguous_cells": int(stats.get("ambiguous", 0)),
        },
        "warnings": warnings,
        "review": review,
        # convenience views (handy for tests / debugging)
        "matrix": matrix,
        "rolls": rolls,
        "method": source,
        "grid_confidence": round(float(grid["confidence"]), 3) if grid else 0.0,
        "ok": True,
    }


def analyze(file_bytes: bytes, *, expected_rows=None, expected_cols=None,
            allow_vlm=None):
    """Read an attendance sheet. OMR-first; falls back to a chunked VLM read
    only when the grid can't be trusted (and a token is available)."""
    if allow_vlm is None:
        allow_vlm = bool(os.environ.get("HF_TOKEN") or
                         os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

    gray, bgr, _rect = rectify(file_bytes)
    gray_e = _enhance(gray)
    preview = preprocess.to_data_uri(gray_e)

    def done(result):
        result.setdefault("processed_image", preview)
        return result

    grid = detect_grid(gray_e, expected_rows, expected_cols)

    warnings = []
    if grid is None or grid["confidence"] < GRID_CONF_MIN:
        warnings.append("grid not detected confidently")
        if allow_vlm:
            return done(_vlm_fallback(bgr, grid, allow_vlm, warnings))
        return done(_fail("Could not detect a table grid. Try a clearer, flatter "
                          "photo or enable High-contrast mode.", grid))

    roles = assign_roles(gray_e, grid["cells"])
    if not roles or not roles["mark_cols"]:
        warnings.append("could not identify mark columns")
        if allow_vlm:
            return done(_vlm_fallback(bgr, grid, allow_vlm, warnings))
        return done(_fail("Detected a grid but no attendance (circle) columns.", grid))

    states, stats = classify_marks(gray_e, grid["cells"], roles["data_rows"], roles["mark_cols"])

    # Optional count gate: if the user told us how many students / days to
    # expect, a mismatch means the grid was mis-segmented -> don't trust OMR.
    count_ok = True
    if expected_rows:
        count_ok = count_ok and len(roles["data_rows"]) == expected_rows
    if expected_cols:
        count_ok = count_ok and len(roles["mark_cols"]) == expected_cols
    if not count_ok:
        warnings.append(
            f"detected {len(roles['data_rows'])}x{len(roles['mark_cols'])} but "
            f"expected {expected_rows or '?'}x{expected_cols or '?'}")

    confident = (stats["spread"] >= SEPARATION_MIN and
                 stats["ambiguous_frac"] <= AMBIGUOUS_FRAC_MAX and
                 stats.get("multi_circle_frac", 0.0) <= 0.05 and
                 grid["confidence"] >= GRID_CONF_MIN and
                 count_ok)

    if (not confident) and allow_vlm:
        warnings.append("low OMR confidence -> VLM fallback")
        return done(_vlm_fallback(bgr, grid, allow_vlm, warnings, roles=roles))

    if not confident:
        warnings.append("low OMR confidence (no token for fallback)")

    rolls, roll_source = read_rolls(bgr, grid["cells"], roles, allow_vlm)
    columns = _columns(roles["mark_cols"])
    result = _assemble(rolls, roll_source, columns, states, roles["data_rows"],
                       roles["mark_cols"], "omr", grid, stats, warnings)
    result["confident"] = bool(confident)
    result["confidence"]["confident"] = bool(confident)
    return done(result)


def _fail(message, grid):
    return {"ok": False, "source": "omr", "error": message, "students": [],
            "columns": [], "dates": [], "matrix": [], "rolls": [],
            "warnings": [message], "review": [],
            "grid_confidence": round(float(grid["confidence"]), 3) if grid else 0.0,
            "confidence": {"grid_ok": grid is not None}}


# --------------------------------------------------------------------------- #
# 8. Chunked VLM fallback (never sends the whole grid at once)                #
# --------------------------------------------------------------------------- #
def _vlm_fallback(bgr, grid, allow_vlm, warnings, roles=None):
    if grid is not None and roles is not None and roles["mark_cols"]:
        return _vlm_fallback_gridded(bgr, grid, roles, warnings)
    # No usable grid geometry: last-resort whole-image read with a tightened,
    # roll-keyed prompt. Small/odd sheets only - large sheets reach the gridded
    # path above and are chunked.
    try:
        env = hf_client.analyze_whole(preprocess.to_data_uri(bgr))
    except Exception as exc:  # noqa: BLE001
        return _fail(f"Vision fallback failed: {exc}", grid)
    env.setdefault("warnings", []).extend(warnings)
    env["source"] = "vlm"
    return env


MAX_CELLS_PER_CALL = 30
MAX_VLM_CALLS = 28
FALLBACK_BUDGET_S = 90      # stay under the gunicorn worker --timeout (120s)


def _vlm_fallback_gridded(bgr, grid, roles, warnings):
    """Tile the sheet into row-band x column-group chunks of at most
    MAX_CELLS_PER_CALL circles, read each with the VLM (dimensions pinned in the
    prompt), and merge by stable (row, col) index - a garbled cell can never
    shift a whole column. Roll numbers come from the separate strip OCR.

    Bounded by both a call cap and a wall-clock budget so a big sheet or a slow
    provider can't blow past the request timeout; cells we never read default to
    'unclear' (a valid partial result), and the user is warned."""
    data_rows = roles["data_rows"]
    mark_cols = roles["mark_cols"]
    cells = grid["cells"]
    columns = _columns(mark_cols)

    rows_per_band = min(len(data_rows), 6)
    cols_per_chunk = max(1, MAX_CELLS_PER_CALL // max(1, rows_per_band))

    rolls, roll_source = read_rolls(bgr, cells, roles, True)
    states = {}
    calls = 0
    start = time.monotonic()
    capped = False
    for rstart in range(0, len(data_rows), rows_per_band):
        if capped:
            break
        band_rows = data_rows[rstart:rstart + rows_per_band]
        for cstart in range(0, len(mark_cols), cols_per_chunk):
            if calls >= MAX_VLM_CALLS or time.monotonic() - start > FALLBACK_BUDGET_S:
                warnings.append("reading budget reached; remaining cells left as unclear")
                capped = True
                break
            band_cols = mark_cols[cstart:cstart + cols_per_chunk]
            uri = _chunk_uri(bgr, cells, band_rows, band_cols)
            if uri is None:
                continue
            try:
                parsed = hf_client.read_chunk(uri, len(band_rows), len(band_cols))
            except Exception:  # noqa: BLE001
                parsed = None
            calls += 1
            for i, r in enumerate(band_rows):
                marks = parsed[i] if parsed and i < len(parsed) else None
                for j, c in enumerate(band_cols):
                    mk = marks[j] if marks and j < len(marks) else "unclear"
                    states[(r, c)] = mk if mk in ("present", "absent", "unclear") else "unclear"

    stats = {"spread": 0.0, "ambiguous": 0, "ambiguous_frac": 0.0}
    result = _assemble(rolls, roll_source, columns, states, data_rows, mark_cols,
                       "vlm", grid, stats, warnings)
    result["confident"] = False
    return result


def _chunk_uri(bgr, cells, band_rows, band_cols):
    """Crop each mark column individually and stitch them side by side, so the
    tile shows EXACTLY len(band_cols) circle columns even when mark_cols skip an
    intervening (non-mark) column - otherwise the VLM sees more columns than the
    prompt declares and its answers shift."""
    y0 = max(0, cells[band_rows[0]][band_cols[0]][1])
    y1 = min(bgr.shape[0], cells[band_rows[-1]][band_cols[0]][3])
    if y1 <= y0:
        return None
    slices = []
    for c in band_cols:
        x0 = max(0, cells[band_rows[0]][c][0])
        x1 = min(bgr.shape[1], cells[band_rows[0]][c][2])
        if x1 > x0:
            sl = bgr[y0:y1, x0:x1]
            if sl.size:
                slices.append(sl)
    if not slices:
        return None
    crop = np.hstack(slices) if len(slices) > 1 else slices[0]
    if max(crop.shape[:2]) > 1400:
        s = 1400 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return preprocess.to_data_uri(crop)
