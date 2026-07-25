"""Google ADK code-first multi-agent composition for the banking workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .agent import run_agent
from .contracts import data_quality_report, load_dataset
from .features import build_customer_features
from .config import SegmentationConfig


def eda_tool(data_path: str) -> dict:
    """Profile a local dataset without sending rows to the model provider."""
    frames = load_dataset(data_path)
    return {"status": "ok", "data_path": data_path, "quality": data_quality_report(frames)}


def feature_engineering_tool(data_path: str) -> dict:
    """Build the local customer feature table and report its schema."""
    frames = load_dataset(data_path)
    features = build_customer_features(frames, SegmentationConfig())
    return {"status": "ok", "customers": len(features), "features": list(features.columns)}


def segmentation_tool(data_path: str, query: str) -> dict:
    """Run the deterministic, audited segmentation workflow locally."""
    result = run_agent(data_path, query, llm_provider="none")
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
        instruction="Route the request to analytics_sequential_pipeline. Use governance_review_loop when explanation or human review is requested. Keep banking records local and return structured summaries.",
        sub_agents=[sequential_pipeline, review_loop],
    )
