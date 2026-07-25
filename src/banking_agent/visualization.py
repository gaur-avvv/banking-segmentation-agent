from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_visualizations(segmented: pd.DataFrame, projection: dict, output_dir: str | Path) -> list[str]:
    """Create deterministic diagnostic charts and return their paths."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    counts = segmented["segment"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#2563eb")
    ax.set_title("Customer segment distribution")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    path = target / "segment_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path))

    points = projection["coordinates"].merge(segmented[["customer_id", "segment"]], on="customer_id", how="left")
    fig, ax = plt.subplots(figsize=(8, 6))
    for segment, group in points.groupby("segment", dropna=False, sort=True):
        ax.scatter(group["pc1"], group["pc2"], s=8, alpha=0.45, label=str(segment))
    ax.set_title("PCA projection of selected customer features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Segment")
    fig.tight_layout()
    path = target / "pca_segments.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path))

    plot_features = [
        ("avg_balance_90d", "Average balance (90d)"),
        ("transaction_frequency_monthly", "Transactions per month"),
        ("recency_days", "Recency (days)"),
    ]
    fig, axes = plt.subplots(1, len(plot_features), figsize=(15, 4))
    for axis, (column, label) in zip(axes, plot_features):
        segmented.boxplot(column=column, by="segment", ax=axis, grid=False, showfliers=False)
        axis.set_title(label)
        axis.set_xlabel("")
        axis.set_ylabel("")
    fig.suptitle("Behavioral features by segment")
    fig.tight_layout()
    path = target / "feature_distributions.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path))
    return paths
