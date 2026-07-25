from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, train_test_split

from .config import SegmentationConfig
from .features import FEATURE_COLUMNS, eligible_feature_matrix


def chronological_split(df: pd.DataFrame, config: SegmentationConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df["as_of_date"].nunique() <= 1:
        # A customer snapshot has no cohort timestamp; split customers reproducibly instead of pretending it is temporal.
        train_val, test = train_test_split(df, test_size=config.test_fraction, random_state=config.random_state)
        validation_fraction_of_remaining = config.validation_fraction / (1 - config.test_fraction)
        train, validation = train_test_split(train_val, test_size=validation_fraction_of_remaining, random_state=config.random_state)
        return train.copy(), validation.copy(), test.copy()
    ordered = df.sort_values(["as_of_date", "customer_id"]).reset_index(drop=True)
    n = len(ordered)
    test_start = max(1, int(n * (1 - config.test_fraction)))
    val_start = max(1, int(n * (1 - config.test_fraction - config.validation_fraction)))
    return ordered.iloc[:val_start], ordered.iloc[val_start:test_start], ordered.iloc[test_start:]


def evaluation_sample(df: pd.DataFrame, config: SegmentationConfig) -> pd.DataFrame:
    if len(df) <= config.max_evaluation_customers:
        return df
    return df.sample(config.max_evaluation_customers, random_state=config.random_state)


def _pipeline(k: int) -> Pipeline:
    # MI uses a leakage-safe proxy target constructed from fold-local behavior ranking.
    prep = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", RobustScaler())])
    return Pipeline([("prep", prep), ("select", SelectKBest(mutual_info_regression, k=k))])


def _behavior_proxy(X: pd.DataFrame) -> np.ndarray:
    balance = X["avg_balance_90d"].fillna(X["avg_balance_90d"].median()).rank(pct=True)
    frequency = X["transaction_frequency_monthly"].rank(pct=True)
    recency = 1 - X["recency_days"].rank(pct=True)
    return (0.5 * balance + 0.3 * frequency + 0.2 * recency).to_numpy()


def evaluate_unsupervised_models(train: pd.DataFrame, validation: pd.DataFrame, config: SegmentationConfig) -> dict:
    X_train, X_val = eligible_feature_matrix(train), eligible_feature_matrix(validation)
    k = min(6, len(FEATURE_COLUMNS))
    transformer = _pipeline(k)
    y_proxy = _behavior_proxy(X_train)
    X_train_t = transformer.fit_transform(X_train, y_proxy)
    X_val_t = transformer.transform(X_val)
    results = {}
    for name, model in {
        "kmeans": KMeans(n_clusters=3, n_init=20, random_state=config.random_state),
        "gmm": GaussianMixture(n_components=3, n_init=10, random_state=config.random_state),
    }.items():
        try:
            labels = model.fit_predict(X_train_t)
            val_labels = model.predict(X_val_t)
            results[name] = {
                "status": "evaluated",
                "validation_silhouette": float(silhouette_score(X_val_t, val_labels)) if len(set(val_labels)) > 1 else -1.0,
                "train_silhouette": float(silhouette_score(X_train_t, labels)),
                "train_davies_bouldin": float(davies_bouldin_score(X_train_t, labels)),
                "model": model, "transformer": transformer,
            }
        except (ValueError, MemoryError) as exc:
            # The deployable rules remain available if an exploratory model is unsuitable.
            results[name] = {"status": "fallback_to_rules", "reason": type(exc).__name__}
    return results


def cross_validate_stability(train: pd.DataFrame, config: SegmentationConfig) -> dict:
    X = eligible_feature_matrix(train)
    scores = []
    folds = min(config.cv_folds, max(2, len(X) // 8))
    for fit_idx, holdout_idx in KFold(n_splits=folds, shuffle=True, random_state=config.random_state).split(X):
        x_fit, x_holdout = X.iloc[fit_idx], X.iloc[holdout_idx]
        trans = _pipeline(min(6, len(FEATURE_COLUMNS)))
        x_fit_t = trans.fit_transform(x_fit, _behavior_proxy(x_fit))
        x_holdout_t = trans.transform(x_holdout)
        labels = KMeans(n_clusters=3, n_init=20, random_state=config.random_state).fit_predict(x_holdout_t)
        if len(set(labels)) > 1:
            scores.append(float(silhouette_score(x_holdout_t, labels)))
    return {"cv_folds": folds, "mean_silhouette": float(np.mean(scores)), "fold_scores": scores}


def derive_rule_thresholds(training_features: pd.DataFrame, config: SegmentationConfig) -> dict:
    """Derive business-rule parameters from training data only; never from validation/test."""
    valid = training_features.loc[~training_features["insufficient_balance_history"]]
    return {
        "priority_balance_threshold": float(valid["avg_balance_90d"].quantile(config.priority_balance_percentile)),
        "priority_frequency_threshold": float(valid["transaction_frequency_monthly"].quantile(config.priority_frequency_percentile)),
        "dormant_days": config.dormant_days,
    }


def assign_rule_segments(features: pd.DataFrame, config: SegmentationConfig, thresholds: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = features.copy()
    thresholds = thresholds or derive_rule_thresholds(features, config)
    balance_threshold = thresholds["priority_balance_threshold"]
    frequency_threshold = thresholds["priority_frequency_threshold"]
    df["segment"] = "regular"
    df["fallback_level"] = 2
    df["assignment_confidence"] = "medium"
    dormant = (df["recency_days"] >= config.dormant_days) | (df["transaction_count_90d"] == 0)
    priority = (df["avg_balance_90d"] >= balance_threshold) & (df["transaction_frequency_monthly"] >= frequency_threshold)
    df.loc[dormant, "segment"] = "dormant"
    df.loc[priority & ~dormant, "segment"] = "priority"
    df.loc[df["insufficient_balance_history"], "segment"] = "needs_review"
    df.loc[dormant | (priority & ~dormant), "fallback_level"] = 1
    df.loc[dormant | (priority & ~dormant), "assignment_confidence"] = "high"
    df.loc[df["insufficient_balance_history"], "fallback_level"] = 3
    df.loc[df["insufficient_balance_history"], "assignment_confidence"] = "low"
    df["segment_reason"] = np.select(
        [df.segment.eq("priority"), df.segment.eq("dormant"), df.segment.eq("needs_review")],
        ["L1: high maintained balance and transaction frequency", "L1: inactive or no recent transactions", "L3: missing balance history; human review required"],
        default="L2: active customer below priority thresholds",
    )
    return df, thresholds


def final_test_report(test: pd.DataFrame, config: SegmentationConfig, thresholds: dict) -> dict:
    # Rules are the selected deployable model; report distribution and no hidden test tuning.
    segmented, _ = assign_rule_segments(test, config, thresholds)
    return {"test_customers": len(test), "segment_distribution": segmented.segment.value_counts().to_dict(), "thresholds": thresholds}
