"""Optional Gemini query planner. The ML workflow remains deterministic without it."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .routing import ALLOWED_INTENTS, route_query


def _load_local_env() -> None:
    """Minimal .env loader; environment variables always take precedence."""
    path = Path.cwd() / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key in {"GEMINI_API_KEY", "GEMINI_MODEL"}:
            os.environ.setdefault(key, value.strip().strip("'\""))


def plan_query_with_gemini(query: str) -> dict:
    """Return a constrained plan; API key is only read from GEMINI_API_KEY at runtime."""
    _load_local_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {**route_query(query), "reason": "GEMINI_API_KEY is not configured."}
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = f'''Classify this retail-bank analytics request into JSON only.
Allowed intents: segment_customers, explain_segment, compare_segments, recommend_conversion.
Request: {query!r}
Return keys intent, requested_outputs, needs_human_input. Do not include customer data.'''
        model = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
        response = client.models.generate_content(model=model, contents=prompt)
        plan = json.loads(response.text)
        if plan.get("intent") not in ALLOWED_INTENTS:
            raise ValueError("Gemini returned an unsupported intent")
        return {"source": "gemini", **plan}
    except Exception as exc:
        return {**route_query(query), "reason": f"Gemini planner unavailable: {type(exc).__name__}"}
