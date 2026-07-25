"""Google ADK code-first multi-agent composition for the banking workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .agent import run_agent
from .contracts import data_quality_report, load_dataset
from .features import build_customer_features
from .config import SegmentationConfig
from .datasets import resolve_dataset_path


def eda_tool(data_path: str | None = None) -> dict:
    """Profile a local dataset without sending rows to the model provider."""
    resolved, reason = resolve_dataset_path(data_path)
    frames = load_dataset(resolved)
    return {"status": "ok", "data_path": resolved, "resolution": reason, "quality": data_quality_report(frames)}


def feature_engineering_tool(data_path: str | None = None) -> dict:
    """Build the local customer feature table and report its schema."""
    resolved, reason = resolve_dataset_path(data_path)
    frames = load_dataset(resolved)
    features = build_customer_features(frames, SegmentationConfig())
    return {"status": "ok", "data_path": resolved, "resolution": reason, "customers": len(features), "features": list(features.columns)}


def segmentation_tool(data_path: str | None = None, query: str = "Segment customers") -> dict:
    """Run the deterministic, audited segmentation workflow locally."""
    resolved, _ = resolve_dataset_path(data_path)
    result = run_agent(resolved, query, llm_provider="none")
    return {"status": "ok", "report": result["report"], "artifacts": result["artifacts"]}


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
    eda_agent = Agent(
        name="eda_agent",
        model=model,
        description="Profiles local banking data and reports quality metrics.",
        instruction="Call eda_tool for dataset profiling. Never request raw rows in your response.",
        tools=[eda_tool],
    )
    feature_agent = Agent(
        name="feature_engineering_agent",
        model=model,
        description="Builds customer-level behavioral features locally.",
        instruction="Call feature_engineering_tool after EDA and summarize engineered columns.",
        tools=[feature_engineering_tool],
    )
    segmentation_agent = Agent(
        name="segmentation_agent",
        model=model,
        description="Runs the audited segmentation and evaluation workflow.",
        instruction="Call segmentation_tool with the user dataset path and query. Do not invent metrics.",
        tools=[segmentation_tool],
    )
    explainability_agent = Agent(
        name="explainability_agent",
        model=model,
        description="Explains segment rules and proposes policy-safe next actions.",
        instruction="Call explainability_tool for each requested segment and mention human review when needed.",
        tools=[explainability_tool],
    )
    review_explainability_agent = Agent(
        name="governance_explainability_agent",
        model=model,
        description="Checks explanations for auditability and human-review needs.",
        instruction="Call explainability_tool and identify any human-review requirement.",
        tools=[explainability_tool],
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
    return Agent(
        name="banking_root_router_agent",
        model=model,
        description="Routes retail banking queries to specialized ADK agents.",
        instruction=(
            "Route every analytics request to analytics_sequential_pipeline. "
            "Use governance_review_loop when explanation, audit, or human review is requested. "
            "The dataset path is the path explicitly provided by the user, otherwise BANKING_DATA_PATH, "
            "otherwise the safe local demo fallback. Never guess banking_data.csv, never ask for the path "
            "again when a fallback is available, and pass the same data_path to every specialist tool. "
            "Keep banking records local, do not expose raw rows or credentials, and return structured summaries."
        ),
        sub_agents=[sequential_pipeline, review_loop],
    )
