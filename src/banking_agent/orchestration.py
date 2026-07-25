from __future__ import annotations

from typing import Any


STEP_REGISTRY = {
    "query_planning": ("orchestrator_agent", "query_planner"),
    "data_validation": ("data_quality_agent", "data_quality_report"),
    "data_cleaning_filtering": ("data_quality_agent", "clean_and_filter_events"),
    "feature_extraction": ("feature_agent", "build_customer_features"),
    "feature_selection": ("feature_agent", "mutual_information_selection"),
    "dimensionality_reduction": ("feature_agent", "pca_projection"),
    "hyperparameter_tuning": ("model_agent", "tune_kmeans_gmm"),
    "fit_diagnostics": ("model_agent", "fit_diagnostics"),
    "leakage_audit": ("governance_agent", "leakage_audit"),
    "model_evaluation": ("model_agent", "cross_validation_and_test_report"),
    "recommendations": ("recommendation_agent", "priority_candidates"),
    "visualization": ("visualization_agent", "create_visualizations"),
    "memory_profile": ("memory_agent", "build_memory_profile"),
    "memory_saved": ("memory_agent", "save_consent_audit"),
}


def build_agent_trace(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate existing workflow events as specialist-agent/tool calls."""
    trace = []
    for index, event in enumerate(events, start=1):
        step = str(event.get("step", "unknown"))
        agent, tool = STEP_REGISTRY.get(step, ("orchestrator_agent", step))
        trace.append({
            "sequence": index,
            "type": "agent_tool_call",
            "agent": agent,
            "tool": tool,
            "step": step,
            "detail": event.get("detail", ""),
            "status": "completed",
        })
    return trace
