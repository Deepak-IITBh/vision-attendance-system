"""Thin client around the Hugging Face Inference router.

We talk to a vision-language model (VLM) through Hugging Face's
OpenAI-compatible **router** endpoint
(``https://router.huggingface.co/v1/chat/completions``). The router picks
whichever inference provider currently serves the requested model.

IMPORTANT design note: we no longer ask the model to read a whole 600-cell
attendance grid in one shot - that is what made it hallucinate. The
deterministic OpenCV reader in :mod:`omr` handles the marks. The model is now
used only for small, reliable sub-tasks:

* :func:`read_roll_strip` - OCR a single narrow column of roll numbers.
* :func:`read_chunk`      - read a small tile of circles (<=~30) for the
  fallback path, with the grid dimensions pinned in the prompt.
* :func:`analyze_whole`   - last-resort whole-image read for sheets where no
  grid could be found at all (kept tightly constrained).
"""

import json
import os
import re

import requests

DEFAULT_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
REQUEST_TIMEOUT = 45  # seconds; chunks are small so this is plenty


def _token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a free token at "
            "https://huggingface.co/settings/tokens and set it as an environment variable."
        )
    return token


def _post(messages, *, max_tokens=300, temperature=0.0, timeout=REQUEST_TIMEOUT,
          retries=1) -> str:
    """POST a chat-completion and return the message text. Retries once on
    timeout / 5xx / 429, then raises RuntimeError."""
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"}

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_err = f"Could not reach Hugging Face: {exc}"
            continue
        if resp.status_code == 200:
            try:
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, ValueError) as exc:
                raise RuntimeError("Unexpected response shape from Hugging Face.") from exc
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            last_err = f"{resp.status_code}: {_error_detail(resp)}"
            continue
        raise RuntimeError(
            f"Hugging Face inference failed ({resp.status_code}): {_error_detail(resp)}")
    raise RuntimeError(last_err or "Hugging Face request failed.")


def vlm_text(prompt: str, data_uri: str, **kw) -> str:
    return _post(
        [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]}],
        **kw,
    )


# --------------------------------------------------------------------------- #
# JSON extraction                                                             #
# --------------------------------------------------------------------------- #
def _strip_to_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
    return text


def _parse_json(text: str):
    try:
        return json.loads(_strip_to_json(text))
    except (json.JSONDecodeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Small sub-tasks used by the OMR pipeline                                     #
# --------------------------------------------------------------------------- #
def read_roll_strip(data_uri: str, n: int):
    """OCR a vertical strip of ``n`` roll numbers. Returns a list of strings
    (best effort) or None."""
    prompt = (
        f"This image is a single vertical column from an attendance sheet "
        f"containing {n} roll numbers, one per row, top to bottom. "
        f"Read each roll number EXACTLY as written (keep digits and letters, "
        f"no extra text). Return ONLY a JSON array of {n} strings in order, "
        f"e.g. [\"1\",\"2\",...]. If a row is unreadable use \"\"."
    )
    data = _parse_json(vlm_text(prompt, data_uri, max_tokens=600))
    if isinstance(data, list):
        return [str(x) for x in data]
    return None


def read_chunk(data_uri: str, n_rows: int, n_cols: int):
    """Read a small tile of attendance circles. Returns a list of ``n_rows``
    lists, each of ``n_cols`` strings in {present, absent, unclear}, or None."""
    prompt = (
        f"This is a small crop of an attendance table with EXACTLY {n_rows} "
        f"row(s) and {n_cols} column(s) of circles. For each circle: a FILLED, "
        f"shaded or scribbled-in circle means \"present\"; an EMPTY/open circle "
        f"(just an outline) means \"absent\"; if you genuinely cannot tell use "
        f"\"unclear\". Return ONLY a JSON array of exactly {n_rows} rows; each "
        f"row is a JSON array of exactly {n_cols} strings, left-to-right, "
        f"top-to-bottom. No prose, no row labels, no roll numbers."
    )
    data = _parse_json(vlm_text(prompt, data_uri, max_tokens=900, timeout=30, retries=0))
    if not isinstance(data, list):
        return None
    # Normalise: ensure list-of-lists.
    if data and not isinstance(data[0], list):
        data = [data]
    # Keep positional alignment: a malformed (non-list) row becomes [] so the
    # merge defaults its cells to "unclear" instead of shifting later rows up.
    return [[str(v).strip().lower() for v in row] if isinstance(row, list) else []
            for row in data]


# --------------------------------------------------------------------------- #
# Last-resort whole-image read (only when no grid could be detected at all)   #
# --------------------------------------------------------------------------- #
_WHOLE_PROMPT = """You are reading a class attendance sheet. Each student row has
a roll number (leftmost) and one circle per date column: a FILLED/shaded/ticked
circle = present, an EMPTY/open circle = absent, illegible = unclear.

Return ONLY a JSON object, no prose, no markdown, exactly this schema:
{
  "dates": ["<column label>", ...],
  "students": [
    { "roll": "<roll number exactly as written>",
      "attendance": { "<column label>": "present" | "absent" | "unclear" } }
  ]
}
Rules: include every row top-to-bottom; do NOT invent rows, columns or roll
numbers; if there is no roll number for a row use its position number; output
valid JSON and nothing else."""


def analyze_whole(data_uri: str) -> dict:
    """Whole-image fallback used only when the grid is undetectable. Returns the
    same roll-keyed envelope shape the OMR path produces."""
    raw = _parse_json(vlm_text(_WHOLE_PROMPT, data_uri, max_tokens=1800, timeout=90))
    if not isinstance(raw, dict) or not isinstance(raw.get("students"), list):
        raise RuntimeError("The model did not return readable JSON for the sheet.")

    dates = [str(d) for d in raw.get("dates", [])] or ["attendance"]
    students, review = [], []
    for i, s in enumerate(raw["students"]):
        if not isinstance(s, dict):
            continue
        roll = str(s.get("roll") or i + 1).strip() or str(i + 1)
        att_in = s.get("attendance") or {}
        attendance = {}
        for d in dates:
            v = str(att_in.get(d, "unclear")).strip().lower()
            v = v if v in ("present", "absent", "unclear") else "unclear"
            attendance[d] = v
            if v == "unclear":
                review.append({"roll": roll, "date": d})
        students.append({"roll": roll, "name": None,
                         "roll_confidence": "low", "attendance": attendance})

    return {
        "source": "vlm", "has_roll_column": True, "roll_source": "ocr",
        "columns": dates, "dates": dates, "students": students,
        "confidence": {"grid_ok": False, "n_rows": len(students), "n_cols": len(dates)},
        "warnings": [], "review": review,
        "matrix": [[st["attendance"][d] for d in dates] for st in students],
        "rolls": [st["roll"] for st in students], "method": "vlm",
        "grid_confidence": 0.0, "ok": True, "confident": False,
    }


def _error_detail(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return err.get("message", json.dumps(err))
            if err:
                return str(err)
        return json.dumps(body)[:300]
    except ValueError:
        return resp.text[:300]
