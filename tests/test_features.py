import pandas as pd

from banking_agent.config import SegmentationConfig
from banking_agent.features import build_customer_features
from banking_agent.modeling import assign_rule_segments


def frames():
    return {
        "customers": pd.DataFrame({"customer_id": ["a", "b"]}),
        "balances": pd.DataFrame({"customer_id": ["a", "a"], "timestamp": ["2025-01-01", "2025-03-31"], "balance": [100, 200]}),
        "transactions": pd.DataFrame({"customer_id": ["a"], "timestamp": ["2025-03-31"], "amount": [30]}),
        "product_holdings": pd.DataFrame({"customer_id": ["a"], "product_name": ["checking"], "status": ["active"]}),
    }


def test_feature_builder_flags_missing_balance_history():
    output = build_customer_features(frames(), SegmentationConfig())
    assert output.loc[output.customer_id.eq("b"), "insufficient_balance_history"].item()


def test_rule_segmenter_does_not_hide_missing_history():
    output, _ = assign_rule_segments(build_customer_features(frames(), SegmentationConfig()), SegmentationConfig())
    assert output.loc[output.customer_id.eq("b"), "segment"].item() == "needs_review"
    assert output.loc[output.customer_id.eq("b"), "fallback_level"].item() == 3
