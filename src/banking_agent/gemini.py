"""Optional hosted query planners with deterministic local fallback."""
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
        if key in {"LLM_PROVIDER", "LLM_MODEL", "BANKING_DATA_PATH", "GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY", "OPEN_AI_API", "OPENAI_API", "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "GROQ_API_KEY", "GROQ_MODEL", "OLLAMA_MODEL", "OLLAMA_BASE_URL"}:
            os.environ.setdefault(key, value.strip().strip("'\""))


def _prompt(query: str) -> str:
    return f'''Classify this retail-bank analytics request into JSON only.
Allowed intents: segment_customers, explain_segment, compare_segments, recommend_conversion.
Request: {query!r}
Return keys intent, requested_outputs, needs_human_input. Do not include customer data.'''


def _parse_plan(text: str, source: str) -> dict:
    plan = json.loads(text)
    if plan.get("intent") not in ALLOWED_INTENTS:
        raise ValueError("provider returned an unsupported intent")
    return {"source": source, **plan}


def _plan_with_gemini(query: str, api_key: str, model: str | None = None) -> dict:
    from google import genai
    client = genai.Client(api_key=api_key)
    model = model or os.environ.get("GEMINI_MODEL", os.environ.get("LLM_MODEL", "gemma-4-26b-a4b-it"))
    response = client.models.generate_content(model=model, contents=_prompt(query))
    return _parse_plan(response.text, f"gemini:{model}")


def _plan_with_openai_compatible(query: str, api_key: str, base_url: str, model: str, source: str) -> dict:
    from openai import OpenAI
    kwargs = {"api_key": api_key}
    kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": _prompt(query)}],
    )
    return _parse_plan(response.choices[0].message.content or "{}", f"{source}:{model}")


def plan_query_with_gemini(query: str, provider: str | None = None, model: str | None = None) -> dict:
    """Use configured Gemini/OpenAI-compatible planning, then deterministic routing."""
    _load_local_env()
    provider = (provider or os.environ.get("LLM_PROVIDER", "auto")).lower()
    model = model or os.environ.get("LLM_MODEL")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_API") or os.environ.get("OPENAI_API")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    attempts = []
    if provider in {"auto", "gemini"} and gemini_key:
        attempts.append(("gemini", lambda: _plan_with_gemini(query, gemini_key, model)))
    if provider in {"auto", "openrouter"} and openrouter_key:
        attempts.append(("openrouter", lambda: _plan_with_openai_compatible(query, openrouter_key, "https://openrouter.ai/api/v1", model or os.environ.get("OPENROUTER_MODEL", "openrouter/free"), "openrouter")))
    if provider in {"auto", "groq"} and groq_key:
        attempts.append(("groq", lambda: _plan_with_openai_compatible(query, groq_key, "https://api.groq.com/openai/v1", model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), "groq")))
    if provider in {"auto", "openai", "openai-compatible"} and openai_key:
        attempts.append(("openai", lambda: _plan_with_openai_compatible(query, openai_key, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "openai-compatible")))
    if provider == "ollama":
        attempts.append(("ollama", lambda: _plan_with_openai_compatible(query, "ollama", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), model or os.environ.get("OLLAMA_MODEL", "gemma3:4b"), "ollama")))
    if provider not in {"auto", "gemini", "openrouter", "groq", "openai", "openai-compatible", "ollama", "none"}:
        return {**route_query(query), "reason": f"Unknown LLM_PROVIDER={provider}; deterministic fallback used."}
    errors = []
    for name, attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            errors.append(f"{name} planner unavailable: {type(exc).__name__}")
    reason = "No provider API key configured." if not attempts else "; ".join(errors)
    return {**route_query(query), "reason": reason}
