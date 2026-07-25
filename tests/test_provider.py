import banking_agent.gemini as planner
from banking_agent.gemini import plan_query_with_gemini


def test_provider_falls_back_without_api_keys(monkeypatch):
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY", "OPEN_AI_API", "OPENAI_API"):
        monkeypatch.delenv(name, raising=False)
    result = plan_query_with_gemini("Which regular customers can be converted to priority?")
    assert result["source"] == "deterministic_router"
    assert result["intent"] == "recommend_conversion"


def test_unknown_provider_is_safe_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown-provider")
    result = plan_query_with_gemini("Compare transaction sizes")
    assert result["source"] == "deterministic_router"
    assert "Unknown LLM_PROVIDER" in result["reason"]


def test_ollama_provider_uses_local_openai_compatible_endpoint(monkeypatch):
    seen = {}

    def fake_plan(query, api_key, base_url, model, source):
        seen.update(api_key=api_key, base_url=base_url, model=model, source=source)
        return {"source": source, "intent": "segment_customers"}

    monkeypatch.setattr(planner, "_plan_with_openai_compatible", fake_plan)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    result = plan_query_with_gemini("Segment customers")
    assert result["source"] == "ollama"
    assert seen["base_url"] == "http://localhost:11434/v1"
    assert seen["api_key"] == "ollama"
    assert seen["model"] == "qwen2.5:3b"
