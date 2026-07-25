"""Optional Google ADK bridge.

The deterministic LangGraph workflow remains the execution engine. When the
optional google-adk package is installed, this module exposes a root ADK agent
descriptor for clients that want ADK discovery/configuration.
"""
from __future__ import annotations

import os


def create_adk_root_agent():
    try:
        from google.adk.agents.llm_agent import Agent
    except ImportError as exc:
        raise RuntimeError("Install the optional ADK extra with: pip install -e '.[adk]'") from exc
    model = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
    return Agent(
        name="banking_orchestrator_agent",
        model=model,
        description="Routes retail-banking analytics requests to specialist agents and tools.",
        instruction="Return structured plans only; never request or expose raw banking records.",
    )


def adk_status() -> dict[str, object]:
    try:
        import google.adk  # noqa: F401
        return {"installed": True, "mode": "optional_adk_bridge"}
    except ImportError:
        return {"installed": False, "mode": "deterministic_langgraph_fallback"}
