"""Gemini client helpers for grading and generation."""

import json
import os
import re
import time

from google import genai
from google.genai import types

from config import FLASH_MODEL, PRO_MODEL

_MAX_RETRIES = 4


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it or add it to a .env file."
        )
    return genai.Client(api_key=api_key)


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return json.loads(text)


def _retry_delay_from_error(exc: Exception) -> float:
    """Parse the suggested retry delay from a 429 error message, default 60s."""
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(match.group(1)) + 2.0 if match else 60.0


def _call_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying up to _MAX_RETRIES times on 429."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                wait = _retry_delay_from_error(exc)
                print(f"[llm] Rate limited — waiting {wait:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    # Final attempt — let it raise naturally
    return fn(*args, **kwargs)


def grade_with_flash(prompt: str) -> dict:
    client = get_client()

    def _call():
        return client.models.generate_content(
            model=FLASH_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

    response = _call_with_retry(_call)
    return _parse_json_response(response.text or "{}")


def generate_with_pro(prompt: str) -> str:
    client = get_client()

    def _call():
        return client.models.generate_content(
            model=PRO_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )

    response = _call_with_retry(_call)
    return (response.text or "").strip()
