from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_demo_data(output_dir: str | Path) -> Path:
    """Create a small non-sensitive dataset for a clone with no external data."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    customer_ids = [f"C{i:04d}" for i in range(1, 121)]
    pd.DataFrame({"customer_id": customer_ids, "open_date": "2022-01-01"}).to_csv(root / "customers.csv", index=False)
    dates = pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")
    balances, transactions, holdings = [], [], []
    for i, customer_id in enumerate(customer_ids):
        tier = i % 3
        base = max(100, [4_000, 24_000, 75_000][tier] + rng.normal(0, 1_500))
        activity = [1, 7, 18][tier]
        for date in dates[::7]:
            balances.append((customer_id, date.isoformat(), max(0, base + rng.normal(0, base * .06))))
        for _ in range(activity * 4):
            date = rng.choice(dates)
            transactions.append((customer_id, pd.Timestamp(date).isoformat(), max(5, rng.lognormal(5.2, .65))))
        holdings.append((customer_id, "checking", "active"))
        if tier == 2:
            holdings.append((customer_id, "rewards_card", "active"))
    pd.DataFrame(balances, columns=["customer_id", "timestamp", "balance"]).to_csv(root / "balances.csv", index=False)
    pd.DataFrame(transactions, columns=["customer_id", "timestamp", "amount"]).to_csv(root / "transactions.csv", index=False)
    pd.DataFrame(holdings, columns=["customer_id", "product_name", "status"]).to_csv(root / "product_holdings.csv", index=False)
    return root
