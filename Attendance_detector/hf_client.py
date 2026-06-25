"""Thin client around the Hugging Face Inference router.

We use a vision-language model (a VLM) through Hugging Face's OpenAI-compatible
**router** endpoint (``https://router.huggingface.co/v1/chat/completions``). The
router automatically picks whichever inference provider currently serves the
requested model, so we don't have to hard-code a provider. The model receives
the preprocessed image plus a strict prompt and returns the attendance table as
JSON.
"""

import json
import os
import re

import requests

DEFAULT_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
REQUEST_TIMEOUT = 90  # seconds

PROMPT = """You are an attendance-sheet reading assistant.

The image is a class attendance sheet. It contains student names and one or more
date columns. Each cell holds a circle (or bubble/checkbox):
- a FILLED / shaded / ticked / crossed circle  -> the student is PRESENT
- an EMPTY / blank / open circle                -> the student is ABSENT

Read the whole sheet carefully and return ONLY a JSON object, no prose, no
markdown fences, using exactly this schema:

{
  "dates": ["<date or column label>", ...],
  "students": [
    {
      "name": "<student name as written>",
      "attendance": { "<date>": "present" | "absent" | "unclear" }
    }
  ]
}

Rules:
- Include every student row you can read, top to bottom.
- If the sheet has a single attendance column with no date, use "attendance" as
  the single key in "dates" and in each student's "attendance" map.
- If a circle is genuinely impossible to judge, use "unclear" for that cell.
- Do not invent students or dates that are not on the sheet.
- Output must be valid JSON and nothing else."""


def _strip_to_json(text: str) -> str:
    """Pull the JSON object out of a model response that may be wrapped in
    markdown fences or surrounded by stray text."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_result(text: str) -> dict:
    candidate = _strip_to_json(text)
    return json.loads(candidate)


def analyze_image(data_uri: str, model: str | None = None):
    """Send the (preprocessed) image to the VLM. Returns (parsed_dict, raw_text).

    Raises RuntimeError with a user-friendly message on configuration or API
    problems, and ValueError if the model reply could not be parsed as JSON.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a free token at "
            "https://huggingface.co/settings/tokens and set it as an environment variable."
        )

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        resp = requests.post(
            ROUTER_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Hugging Face: {exc}") from exc

    if resp.status_code != 200:
        detail = _error_detail(resp)
        raise RuntimeError(
            f"Hugging Face inference failed ({resp.status_code}): {detail}"
        )

    try:
        raw = resp.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError("Unexpected response shape from Hugging Face.") from exc

    try:
        parsed = parse_result(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "The model did not return readable JSON. Try a clearer photo or the "
            "binarize option."
        ) from exc

    return parsed, raw


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
