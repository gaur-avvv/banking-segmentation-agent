import pandas as pd

from banking_agent.visualization import create_visualizations


def test_visualizations_create_expected_diagnostic_artifacts(tmp_path):
    segmented = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "segment": ["priority", "regular", "dormant", "regular"],
            "avg_balance_90d": [100.0, 50.0, 0.0, 60.0],
            "transaction_frequency_monthly": [5.0, 2.0, 0.0, 3.0],
            "recency_days": [1, 5, 100, 10],
        }
    )
    projection = {"coordinates": pd.DataFrame({"customer_id": ["a", "b", "c", "d"], "pc1": [0, 1, 2, 3], "pc2": [3, 2, 1, 0]})}
    paths = create_visualizations(segmented, projection, tmp_path)
    assert {path.rsplit("/", 1)[-1] for path in paths} == {
        "segment_distribution.png",
        "pca_segments.png",
        "feature_distributions.png",
    }
    assert all(tmp_path.joinpath(path.rsplit("/", 1)[-1]).exists() for path in paths)
