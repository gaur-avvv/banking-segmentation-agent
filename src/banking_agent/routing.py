"""Deterministic query routing with a safe, inspectable fallback."""
from __future__ import annotations

ALLOWED_INTENTS = {"segment_customers", "explain_segment", "compare_segments", "recommend_conversion"}


def route_query(query: str) -> dict:
    normalized = " ".join(query.lower().split())
    rules = (
        ("recommend_conversion", ("convert", "conversion", "candidate", "next best", "recommend")),
        ("compare_segments", ("compare", "average", "median", "transaction size")),
        ("explain_segment", ("why", "basis", "explain", "selected")),
        ("segment_customers", ("segment", "priority", "regular", "dormant")),
    )
    for intent, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return {"intent": intent, "source": "deterministic_router", "needs_human_input": False}
    return {
        "intent": "segment_customers",
        "source": "deterministic_router_fallback",
        "needs_human_input": True,
        "reason": "No supported intent was unambiguously detected; defaulted to segmentation.",
    }
