from __future__ import annotations

import pandas as pd


def priority_candidates(segmented: pd.DataFrame, thresholds: dict, limit: int = 25) -> pd.DataFrame:
    candidates = segmented.loc[segmented.segment.eq("regular")].copy()
    balance_gap = (thresholds["priority_balance_threshold"] - candidates["avg_balance_90d"]).clip(lower=0)
    frequency_gap = (thresholds["priority_frequency_threshold"] - candidates["transaction_frequency_monthly"]).clip(lower=0)
    candidates["priority_gap_score"] = balance_gap / max(thresholds["priority_balance_threshold"], 1) + frequency_gap / max(thresholds["priority_frequency_threshold"], 1)
    candidates["recommended_action"] = ""
    candidates.loc[balance_gap > frequency_gap, "recommended_action"] = "Offer a savings goal or payroll balance-maintenance journey."
    candidates.loc[frequency_gap >= balance_gap, "recommended_action"] = "Offer autopay, debit-card activation, or rewards onboarding."
    candidates["recommendation_reason"] = "Close to priority threshold; recommendation targets the largest remaining behavior gap."
    return candidates.sort_values("priority_gap_score").head(limit)
