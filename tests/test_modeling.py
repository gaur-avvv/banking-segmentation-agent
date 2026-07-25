import numpy as np
import pandas as pd

from banking_agent.config import SegmentationConfig
from banking_agent.modeling import evaluate_unsupervised_models, leakage_audit


def _features(n=30):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "avg_balance_90d": rng.normal(1000, 100, n),
            "min_balance_90d": rng.normal(800, 100, n),
            "balance_stability_90d": rng.uniform(10, 50, n),
            "transaction_count_90d": rng.integers(1, 30, n),
            "transaction_frequency_monthly": rng.uniform(1, 10, n),
            "avg_transaction_amount": rng.uniform(20, 500, n),
            "median_transaction_amount": rng.uniform(20, 500, n),
            "recency_days": rng.integers(1, 60, n),
            "active_product_count": rng.integers(0, 4, n),
            "insufficient_balance_history": False,
            "as_of_date": "2025-01-01",
        }
    )


def test_auto_tuning_returns_fit_diagnostics():
    features = _features()
    scores = evaluate_unsupervised_models(features.iloc[:24], features.iloc[24:], SegmentationConfig())
    assert scores["kmeans"]["status"] == "tuned"
    assert "best_params" in scores["kmeans"]
    assert scores["kmeans"]["fit_check"]["status"] in {"acceptable_generalization", "overfitting_risk", "underfitting_risk"}


def test_leakage_audit_detects_customer_overlap():
    features = _features(6)
    report = {"threshold_source": "training_partition_only"}
    audit = leakage_audit(features.iloc[:3], features.iloc[2:5], features.iloc[5:], report)
    assert audit["status"] == "failed"
    assert audit["overlap_counts"]["train_validation"] == 1
