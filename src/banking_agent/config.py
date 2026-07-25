from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationConfig:
    lookback_days: int = 90
    dormant_days: int = 90
    priority_balance_percentile: float = 0.80
    priority_frequency_percentile: float = 0.70
    random_state: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.20
    cv_folds: int = 5
    min_segment_size: int = 10
    max_evaluation_customers: int = 25_000
