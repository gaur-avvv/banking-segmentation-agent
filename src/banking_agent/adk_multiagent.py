"""Google ADK code-first multi-agent composition for the banking workflow."""
from __future__ import annotations

import json
import os
import copy
import threading
import time
from pathlib import Path

from .agent import run_agent
from .contracts import data_quality_report, load_dataset
from .features import build_customer_features
from .config import SegmentationConfig
from .datasets import resolve_dataset_path


_TOOL_CACHE: dict[tuple[str, str, str], tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = int(os.environ.get("BANKING_AGENT_CACHE_TTL", "900"))


def _cached_tool(kind: str, resolved: str, query: str, producer) -> dict:
    key = (kind, resolved, query)
    now = time.monotonic()
    with _CACHE_LOCK:
        item = _TOOL_CACHE.get(key)
        if item and now - item[0] < _CACHE_TTL_SECONDS:
            result = copy.deepcopy(item[1])
            result["cache_hit"] = True
            return result
    result = producer()
    with _CACHE_LOCK:
        _TOOL_CACHE[key] = (now, copy.deepcopy(result))
    result["cache_hit"] = False
    return result


def eda_tool(data_path: str | None = None) -> dict:
    """Profile a local dataset without sending rows to the model provider."""
    resolved, reason = resolve_dataset_path(data_path)
    return _cached_tool("eda", resolved, "", lambda: {"status": "ok", "data_path": resolved, "resolution": reason, "quality": data_quality_report(load_dataset(resolved))})


def feature_engineering_tool(data_path: str | None = None) -> dict:
    """Build the local customer feature table and report its schema."""
    resolved, reason = resolve_dataset_path(data_path)
    return _cached_tool("features", resolved, "", lambda: _feature_result(resolved, reason))


def _feature_result(resolved: str, reason: str) -> dict:
    features = build_customer_features(load_dataset(resolved), SegmentationConfig())
    return {"status": "ok", "data_path": resolved, "resolution": reason, "customers": len(features), "features": list(features.columns)}


def segmentation_tool(data_path: str | None = None, query: str = "Segment customers") -> dict:
    """Run the deterministic, audited segmentation workflow locally."""
    resolved, _ = resolve_dataset_path(data_path)
    return _cached_tool("segmentation", resolved, query.strip().lower(), lambda: _segmentation_result(resolved, query))


def _segmentation_result(resolved: str, query: str) -> dict:
    result = run_agent(resolved, query, llm_provider="none")
    return {"status": "ok", "report": result["report"], "artifacts": result["artifacts"]}


def query_route_tool(query: str) -> dict:
    """Choose the smallest specialist route for a natural-language request."""
    text = query.lower()
    governance = any(word in text for word in ("explain", "why", "audit", "review", "basis"))
    analytics = any(word in text for word in ("segment", "compare", "average", "transaction", "convert", "recommend"))
    if governance and not analytics:
        return {"route": "governance_review_loop", "agents": ["governance_explainability_agent"], "reason": "explanation/audit intent"}
    if analytics:
        return {"route": "analytics_sequential_pipeline", "agents": ["eda_agent", "feature_engineering_agent", "segmentation_agent", "explainability_agent"], "reason": "data/segmentation intent"}
    return {"route": "analytics_sequential_pipeline", "agents": ["eda_agent", "feature_engineering_agent"], "reason": "safe exploratory fallback"}


def explainability_tool(segment: str) -> dict:
    """Return policy-safe explanations and next actions for a segment."""
    profiles = {
        "priority": {"basis": "High maintained balance and transaction frequency.", "action": "Review relationship expansion and savings/rewards opportunities."},
        "regular": {"basis": "Active behavior below the training-derived priority thresholds.", "action": "Consider balance-building and engagement actions."},
        "dormant": {"basis": "Low recent activity or no recent transactions.", "action": "Use compliant re-engagement and service assistance."},
        "needs_review": {"basis": "Insufficient balance history for a confident assignment.", "action": "Route to human review before personalization."},
    }
    return profiles.get(segment.lower(), {"basis": "Unknown segment.", "action": "Request clarification."})


def create_multi_agent_root_agent():
    """Build root router + sequential and loop specialist agents using ADK."""
    try:
        from google.adk.agents import Agent, LoopAgent, SequentialAgent
    except ImportError as exc:
        raise RuntimeError("Install the optional ADK extra with: pip install -e '.[adk]'") from exc

    model = os.environ.get("ADK_MODEL", os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it"))
    try:
        from google.genai.types import GenerateContentConfig
        generation_config = GenerateContentConfig(temperature=0.1, max_output_tokens=512)
    except ImportError:
        generation_config = None
    try:
        from google.adk.models import Gemini
        from google.genai.types import HttpRetryOptions
        model_ref = Gemini(model=model, retry_options=HttpRetryOptions(initial_delay=1, attempts=3))
    except ImportError:
        model_ref = model
    common = {"generate_content_config": generation_config} if generation_config is not None else {}
    eda_agent = Agent(
        name="eda_agent",
        model=model_ref,
        description="Profiles local banking data and reports quality metrics.",
        instruction="Call eda_tool for dataset profiling. Never request raw rows in your response.",
        tools=[eda_tool], **common,
    )
    feature_agent = Agent(
        name="feature_engineering_agent",
        model=model_ref,
        description="Builds customer-level behavioral features locally.",
        instruction="Call feature_engineering_tool after EDA and summarize engineered columns.",
        tools=[feature_engineering_tool], **common,
    )
    segmentation_agent = Agent(
        name="segmentation_agent",
        model=model_ref,
        description="Runs the audited segmentation and evaluation workflow.",
        instruction="Call segmentation_tool with the user dataset path and query. Do not invent metrics.",
        tools=[segmentation_tool], **common,
    )
    explainability_agent = Agent(
        name="explainability_agent",
        model=model_ref,
        description="Explains segment rules and proposes policy-safe next actions.",
        instruction="Call explainability_tool for each requested segment and mention human review when needed.",
        tools=[explainability_tool], **common,
    )
    review_explainability_agent = Agent(
        name="governance_explainability_agent",
        model=model_ref,
        description="Checks explanations for auditability and human-review needs.",
        instruction="Call explainability_tool and identify any human-review requirement.",
        tools=[explainability_tool], **common,
    )
    sequential_pipeline = SequentialAgent(
        name="analytics_sequential_pipeline",
        description="Runs EDA, feature engineering, segmentation, and explanation in order.",
        sub_agents=[eda_agent, feature_agent, segmentation_agent, explainability_agent],
    )
    review_loop = LoopAgent(
        name="governance_review_loop",
        description="Repeats governance review once to ensure the response is auditable.",
        sub_agents=[review_explainability_agent],
        max_iterations=1,
    )
    if os.environ.get("ADK_STABLE_MODE", "0").lower() in {"1", "true", "yes"}:
        return Agent(
            name="banking_root_router_agent",
            model=model_ref,
            description="Stable single-instruction banking router for uninterrupted ADK chat.",
            instruction=(
                "Use query_route_tool first, then call only the required local tools. "
                "For analytics call eda_tool, feature_engineering_tool, and segmentation_tool as needed. "
                "For explanations call explainability_tool. Keep all records local, return concise structured summaries, "
                "and never expose credentials or private chain-of-thought. Reuse cache_hit results."
            ),
            tools=[query_route_tool, eda_tool, feature_engineering_tool, segmentation_tool, explainability_tool],
            **common,
        )
    return Agent(
        name="banking_root_router_agent",
        model=model_ref,
        description="Routes retail banking queries to specialized ADK agents.",
        instruction=(
            "First call query_route_tool to select the smallest specialist route dynamically. "
            "Segmentation/comparison/recommendation uses analytics_sequential_pipeline; "
            "explanation, audit, or human review uses governance_review_loop (and analytics only if needed). "
            "The dataset path is the path explicitly provided by the user, otherwise BANKING_DATA_PATH, "
            "otherwise the safe local demo fallback. Never guess banking_data.csv, never ask for the path "
            "again when a fallback is available, and pass the same data_path to every specialist tool. "
            "Keep banking records local, do not expose raw rows or credentials, and return concise structured summaries. "
            "Reuse tool results when cache_hit is true; do not call the same tool twice for an unchanged path/query."
        ),
        **common,
        tools=[query_route_tool],
        sub_agents=[sequential_pipeline, review_loop],
    )


def create_dynamic_agent(query: str):
    """Create a query-scoped ADK root with only the selected route enabled."""
    root = create_multi_agent_root_agent()
    route = query_route_tool(query)["route"]
    root.sub_agents = [agent for agent in root.sub_agents if agent.name == route]
    return root
