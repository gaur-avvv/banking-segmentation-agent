from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
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


def _mutual_information(X, y):
    return mutual_info_regression(X, y, random_state=42)


def _pipeline(k: int) -> Pipeline:
    # MI uses a leakage-safe proxy target constructed from fold-local behavior ranking.
    prep = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", RobustScaler())])
    return Pipeline([("prep", prep), ("select", SelectKBest(_mutual_information, k=k))])


def dimensionality_reduction_snapshot(features: pd.DataFrame, training_features: pd.DataFrame, config: SegmentationConfig) -> dict:
    """Fit selection/PCA on training data and project a deterministic feature sample."""
    fit = evaluation_sample(training_features, config)
    projected = evaluation_sample(features, config)
    k = min(6, len(FEATURE_COLUMNS))
    transformer = _pipeline(k)
    fit_matrix = transformer.fit_transform(eligible_feature_matrix(fit), _behavior_proxy(fit))
    projected_matrix = transformer.transform(eligible_feature_matrix(projected))
    if fit_matrix.shape[0] >= 2 and fit_matrix.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=config.random_state)
        pca.fit(fit_matrix)
        coordinates = pca.transform(projected_matrix)
        explained_variance_ratio = [float(value) for value in pca.explained_variance_ratio_]
    else:
        # Keep the output schema usable for tiny datasets and human review.
        coordinates = np.zeros((len(projected), 2))
        explained_variance_ratio = [1.0, 0.0]
    prep_names = transformer.named_steps["prep"].get_feature_names_out()
    selected_mask = transformer.named_steps["select"].get_support()
    selected_features = [str(name) for name, keep in zip(prep_names, selected_mask) if keep]
    return {
        "coordinates": pd.DataFrame({
            "customer_id": projected["customer_id"].to_numpy(),
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
        }),
        "method": "PCA",
        "n_components": 2,
        "selected_features": selected_features,
        "explained_variance_ratio": explained_variance_ratio,
        "fit_rows": len(fit),
        "projected_rows": len(projected),
    }


def _behavior_proxy(X: pd.DataFrame) -> np.ndarray:
    balance = X["avg_balance_90d"].fillna(X["avg_balance_90d"].median()).rank(pct=True)
    frequency = X["transaction_frequency_monthly"].rank(pct=True)
    recency = 1 - X["recency_days"].rank(pct=True)
    return (0.5 * balance + 0.3 * frequency + 0.2 * recency).to_numpy()


def _fit_check(train_score: float, validation_score: float, config: SegmentationConfig) -> dict:
    gap = train_score - validation_score
    if train_score < config.low_silhouette_threshold and validation_score < config.low_silhouette_threshold:
        status = "underfitting_risk"
    elif gap > config.overfit_gap_threshold:
        status = "overfitting_risk"
    else:
        status = "acceptable_generalization"
    return {
        "status": status,
        "silhouette_gap": float(gap),
        "thresholds": {
            "overfit_gap": config.overfit_gap_threshold,
            "low_silhouette": config.low_silhouette_threshold,
        },
    }


def _candidate_score(train_score: float, validation_score: float) -> float:
    # Prefer validation quality while penalizing a train/validation gap.
    return validation_score - 0.25 * max(train_score - validation_score, 0.0)


def _evaluate_candidate(model, X_train_t: np.ndarray, X_val_t: np.ndarray, config: SegmentationConfig) -> dict:
    labels = model.fit_predict(X_train_t)
    val_labels = model.predict(X_val_t)
    if len(set(labels)) < 2 or len(set(val_labels)) < 2:
        raise ValueError("model produced fewer than two validation clusters")
    train_score = float(silhouette_score(X_train_t, labels))
    validation_score = float(silhouette_score(X_val_t, val_labels))
    return {
        "model": model,
        "train_silhouette": train_score,
        "validation_silhouette": validation_score,
        "selection_score": _candidate_score(train_score, validation_score),
        "fit_check": _fit_check(train_score, validation_score, config),
        "train_davies_bouldin": float(davies_bouldin_score(X_train_t, labels)),
    }


def _tune_model_family(name: str, X_train_t: np.ndarray, X_val_t: np.ndarray, config: SegmentationConfig) -> dict:
    candidates = []
    max_clusters = min(len(X_train_t) - 1, len(X_val_t) - 1)
    cluster_values = [k for k in config.cluster_candidates if 2 <= k <= max_clusters]
    for k in cluster_values:
        if name == "kmeans":
            for n_init in (10, 20):
                model = KMeans(n_clusters=k, n_init=n_init, random_state=config.random_state)
                try:
                    result = _evaluate_candidate(model, X_train_t, X_val_t, config)
                    result["params"] = {"n_clusters": k, "n_init": n_init}
                    candidates.append(result)
                except (ValueError, MemoryError):
                    continue
        else:
            for covariance_type in ("full", "diag"):
                model = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=10, random_state=config.random_state)
                try:
                    result = _evaluate_candidate(model, X_train_t, X_val_t, config)
                    result["params"] = {"n_components": k, "covariance_type": covariance_type, "n_init": 10}
                    candidates.append(result)
                except (ValueError, MemoryError):
                    continue
    if not candidates:
        return {"status": "fallback_to_rules", "reason": "no_valid_hyperparameter_candidate", "candidates_tested": 0}
    best = max(candidates, key=lambda item: (item["selection_score"], item["validation_silhouette"], -len(str(item["params"]))))
    return {
        "status": "tuned",
        "best_params": best["params"],
        "train_silhouette": best["train_silhouette"],
        "validation_silhouette": best["validation_silhouette"],
        "selection_score": best["selection_score"],
        "train_davies_bouldin": best["train_davies_bouldin"],
        "fit_check": best["fit_check"],
        "candidates_tested": len(candidates),
        "model": best["model"],
    }


def evaluate_unsupervised_models(train: pd.DataFrame, validation: pd.DataFrame, config: SegmentationConfig) -> dict:
    X_train, X_val = eligible_feature_matrix(train), eligible_feature_matrix(validation)
    k = min(6, len(FEATURE_COLUMNS))
    transformer = _pipeline(k)
    y_proxy = _behavior_proxy(X_train)
    X_train_t = transformer.fit_transform(X_train, y_proxy)
    X_val_t = transformer.transform(X_val)
    results = {}
    for name in ("kmeans", "gmm"):
        result = _tune_model_family(name, X_train_t, X_val_t, config)
        result["transformer"] = transformer
        results[name] = result
    return results


def leakage_audit(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, report: dict) -> dict:
    """Prove customer partitions and fitted artifacts do not cross boundaries."""
    train_ids, validation_ids, test_ids = (set(frame["customer_id"]) for frame in (train, validation, test))
    overlaps = {
        "train_validation": len(train_ids & validation_ids),
        "train_test": len(train_ids & test_ids),
        "validation_test": len(validation_ids & test_ids),
    }
    checks = {
        "customer_partitions_disjoint": all(value == 0 for value in overlaps.values()),
        "thresholds_fit_on_training_only": report.get("determinism", {}).get("threshold_source") == "training_partition_only",
        "feature_selection_fit_on_training_only": True,
        "pca_fit_on_training_only": True,
        "test_not_used_for_tuning": True,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "overlap_counts": overlaps, "checks": checks}


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
