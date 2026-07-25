from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SegmentationConfig

FEATURE_COLUMNS = [
    "avg_balance_90d", "min_balance_90d", "balance_stability_90d",
    "transaction_count_90d", "transaction_frequency_monthly",
    "avg_transaction_amount", "median_transaction_amount", "recency_days",
    "active_product_count",
]


def _clean_events(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {name: df.copy() for name, df in frames.items()}
    for name in ("balances", "transactions"):
        df = result[name]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df.dropna(subset=["customer_id", "timestamp"], inplace=True)
        df.drop_duplicates(subset=["customer_id", "timestamp"], inplace=True)
    result["balances"]["balance"] = pd.to_numeric(result["balances"]["balance"], errors="coerce")
    result["transactions"]["amount"] = pd.to_numeric(result["transactions"]["amount"], errors="coerce")
    return result


def build_customer_features(frames: dict[str, pd.DataFrame], config: SegmentationConfig) -> pd.DataFrame:
    frames = _clean_events(frames)
    customers = frames["customers"][["customer_id"]].drop_duplicates().copy()
    latest = max(frames["balances"]["timestamp"].max(), frames["transactions"]["timestamp"].max())
    cutoff = latest - pd.Timedelta(days=config.lookback_days)
    balances = frames["balances"].query("timestamp >= @cutoff")
    txns = frames["transactions"].query("timestamp >= @cutoff")
    balance_features = balances.groupby("customer_id").agg(
        avg_balance_90d=("balance", "mean"), min_balance_90d=("balance", "min"),
        balance_stability_90d=("balance", "std"),
    )
    txn_features = txns.groupby("customer_id").agg(
        transaction_count_90d=("amount", "size"), avg_transaction_amount=("amount", "mean"),
        median_transaction_amount=("amount", "median"), last_transaction=("timestamp", "max"),
    )
    txn_features["transaction_frequency_monthly"] = txn_features["transaction_count_90d"] * 30 / config.lookback_days
    txn_features["recency_days"] = (latest - txn_features.pop("last_transaction")).dt.days
    holdings = frames["product_holdings"].query("status == 'active'").groupby("customer_id").size().rename("active_product_count")
    features = customers.join(balance_features, on="customer_id").join(txn_features, on="customer_id").join(holdings, on="customer_id")
    features["as_of_date"] = latest.date().isoformat()
    # Semantic defaults: no activity/product means zero; unobserved balance is preserved as a review flag.
    for column in ["transaction_count_90d", "transaction_frequency_monthly", "active_product_count"]:
        features[column] = features[column].fillna(0)
    features["recency_days"] = features["recency_days"].fillna(config.lookback_days + 1)
    features["insufficient_balance_history"] = features["avg_balance_90d"].isna()
    return features


def eligible_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
