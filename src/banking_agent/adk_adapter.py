"""Optional Google ADK bridge.

The deterministic LangGraph workflow remains the execution engine. When the
optional google-adk package is installed, this module exposes a root ADK agent
descriptor for clients that want ADK discovery/configuration.
"""
from __future__ import annotations

import os


def create_adk_root_agent():
    from .adk_multiagent import create_multi_agent_root_agent
    return create_multi_agent_root_agent()


def adk_status() -> dict[str, object]:
    try:
        import google.adk  # noqa: F401
        return {"installed": True, "mode": "optional_adk_bridge"}
    except ImportError:
        return {"installed": False, "mode": "deterministic_langgraph_fallback"}
