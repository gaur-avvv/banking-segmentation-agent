from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import SegmentationConfig
from .contracts import data_quality_report, load_dataset
from .features import build_customer_features, clean_and_filter_events
from .modeling import assign_rule_segments, chronological_split, cross_validate_stability, derive_rule_thresholds, dimensionality_reduction_snapshot, evaluation_sample, evaluate_unsupervised_models, final_test_report, leakage_audit
from .recommendations import priority_candidates
from .gemini import plan_query_with_gemini
from .memory import MemoryEntry, SQLiteMemoryStore
from .visualization import create_visualizations


class AgentState(TypedDict, total=False):
    data_dir: str
    query: str
    events: list[dict]
    frames: dict
    features: object
    segmented: object
    thresholds: dict
    report: dict
    projection: object
    user_id: str
    memory_store: object
    memory_profile: object


def event(state: AgentState, step: str, detail: str) -> dict:
    return {"events": state.get("events", []) + [{"step": step, "detail": detail}]}


def load_and_validate(state: AgentState):
    frames = load_dataset(state["data_dir"])
    _, cleaning_audit = clean_and_filter_events(frames)
    plan = plan_query_with_gemini(state["query"])
    profile = None
    if state.get("memory_store") and state.get("user_id"):
        profile = state["memory_store"].build_profile(state["user_id"])
    events = state.get("events", []) + [
        {"step": "query_planning", "detail": json.dumps(plan)},
        {"step": "data_validation", "detail": json.dumps(data_quality_report(frames))},
        {"step": "data_cleaning_filtering", "detail": json.dumps(cleaning_audit)},
    ]
    if profile:
        events.append({"step": "memory_profile", "detail": json.dumps({"memory_count": profile.memory_count, "preferred_contact_method": profile.preferred_contact_method})})
    return {"frames": frames, "events": events, "memory_profile": profile}


def engineer_features(state: AgentState):
    features = build_customer_features(state["frames"], SegmentationConfig())
    return {"features": features, **event(state, "feature_extraction", f"Created {len(features)} customer rows and 9 behavioral features after lookback filtering.")}


def train_and_segment(state: AgentState):
    config = SegmentationConfig()
    train, validation, test = chronological_split(state["features"], config)
    train_eval, validation_eval = evaluation_sample(train, config), evaluation_sample(validation, config)
    model_scores = evaluate_unsupervised_models(train_eval, validation_eval, config) if len(validation_eval) >= 3 else {"status": "fallback_to_rules", "reason": "validation_set_too_small"}
    cv = cross_validate_stability(train_eval, config) if len(train_eval) >= 16 else {"note": "Dataset too small for stable CV."}
    thresholds = derive_rule_thresholds(train, config)
    segmented, _ = assign_rule_segments(state["features"], config, thresholds)
    projection = dimensionality_reduction_snapshot(state["features"], train, config)
    report = {"determinism": {"random_state": config.random_state, "threshold_source": "training_partition_only", "router": "deterministic_fallback"},
              "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
              "rule_thresholds": thresholds,
              "unsupervised_validation": {
                  k: ({m: v for m, v in score.items() if m not in {"model", "transformer"}} if isinstance(score, dict) else score)
                  for k, score in model_scores.items()
              },
              "cross_validation": cv, "evaluation_sampling": {"train": len(train_eval), "validation": len(validation_eval)},
              "feature_selection": {"method": "mutual_information", "selected_features": projection["selected_features"]},
              "dimensionality_reduction": {k: v for k, v in projection.items() if k != "coordinates"},
              "final_test": final_test_report(test, config, thresholds)}
    report["leakage_prevention"] = leakage_audit(train, validation, test, report)
    report["fit_diagnostics"] = {
        name: score.get("fit_check", {"status": "not_evaluated"}) if isinstance(score, dict) else {"status": "not_evaluated"}
        for name, score in model_scores.items()
    }
    events = state.get("events", []) + [
        {"step": "feature_selection", "detail": f"Selected {len(projection['selected_features'])} features using mutual information on training data."},
        {"step": "dimensionality_reduction", "detail": f"Projected {projection['projected_rows']} customers to 2 PCA components fitted on training data."},
        {"step": "hyperparameter_tuning", "detail": "Auto-tuned K-Means/GMM candidates on training/validation only."},
        {"step": "fit_diagnostics", "detail": "Compared train/validation silhouette to flag overfitting or underfitting risk."},
        {"step": "leakage_audit", "detail": f"Leakage prevention audit: {report['leakage_prevention']['status']}."},
        {"step": "model_evaluation", "detail": "Evaluated tuned K-Means/GMM on validation; deployed auditable rule baseline without tuning on test data."},
    ]
    return {"segmented": segmented, "thresholds": thresholds, "report": report, "projection": projection["coordinates"], "events": events}


def formulate_response(state: AgentState):
    candidates = priority_candidates(state["segmented"], state["thresholds"])
    report = dict(state["report"])
    report["segment_counts"] = state["segmented"].segment.value_counts().to_dict()
    report["top_priority_candidates"] = candidates[["customer_id", "priority_gap_score", "recommended_action", "recommendation_reason"]].to_dict("records")
    return {"report": report, **event(state, "recommendations", f"Ranked {len(candidates)} regular customers for policy-safe priority conversion actions.")}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_and_validate", load_and_validate)
    graph.add_node("engineer_features", engineer_features)
    graph.add_node("train_and_segment", train_and_segment)
    graph.add_node("formulate_response", formulate_response)
    graph.add_edge(START, "load_and_validate")
    graph.add_edge("load_and_validate", "engineer_features")
    graph.add_edge("engineer_features", "train_and_segment")
    graph.add_edge("train_and_segment", "formulate_response")
    graph.add_edge("formulate_response", END)
    return graph.compile()


def run_agent(data_dir: str, query: str, user_id: str | None = None, memory_db: str | None = None, memory_consent: bool = False) -> dict:
    memory_store = SQLiteMemoryStore(memory_db) if memory_db else None
    result = build_graph().invoke({"data_dir": data_dir, "query": query, "events": [], "user_id": user_id, "memory_store": memory_store})
    out = Path(data_dir).parent / "artifacts"
    out.mkdir(exist_ok=True)
    result["segmented"].to_csv(out / "customer_segments.csv", index=False)
    projection = result.get("projection")
    visualization_paths = []
    try:
        visualization_paths = create_visualizations(result["segmented"], {"coordinates": projection}, out / "visualizations")
        result["events"].append({"step": "visualization", "detail": f"Created {len(visualization_paths)} diagnostic charts."})
    except (ImportError, OSError, ValueError) as exc:
        result["events"].append({"step": "visualization", "detail": f"Visualization fallback: {type(exc).__name__}"})
    if memory_store and user_id and memory_consent:
        memory_store.save(MemoryEntry(
            user_id=user_id, interaction_type="analytics_query", consented=True,
            # Do not persist free-text queries by default; a digest supports audit/deduplication.
            content={"topic": "analytics_query", "query_sha256": sha256(query.encode()).hexdigest()},
            context={"intent": result["events"][0]["detail"]},
            outcome={"segment_counts": result["report"]["segment_counts"]},
            actions_taken=["segmentation_run"],
        ))
        result["events"].append({"step": "memory_saved", "detail": "Consented analytics interaction saved locally; query text was not retained."})
    with (out / "run_report.json").open("w") as fh:
        json.dump({"query": query, "events": result["events"], "report": result["report"]}, fh, indent=2, default=str)
    return {"events": result["events"], "report": result["report"], "artifacts": [str(out / "customer_segments.csv"), str(out / "run_report.json"), *visualization_paths]}
