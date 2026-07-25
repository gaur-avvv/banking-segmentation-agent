"""Prepare the supplied bank transaction ZIP for local, leakage-safe experiments.

The raw source is retained locally as data/bank_transactions.csv. The three
transaction-level splits are customer-disjoint, so a customer's history never
appears in more than one split. The agent still derives its deployable rule
thresholds from the training customer-feature partition only.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_COLUMNS = [
    "TransactionID",
    "CustomerID",
    "CustomerDOB",
    "CustGender",
    "CustLocation",
    "CustAccountBalance",
    "TransactionDate",
    "TransactionTime",
    "TransactionAmount (INR)",
]


def default_source() -> Path:
    configured = os.environ.get("BANK_TRANSACTIONS_ZIP")
    candidates = [
        Path(configured) if configured else None,
        Path(r"C:\Users\Dell\Downloads\Compressed\bank_transactions.csv.zip"),
        Path("/mnt/c/Users/Dell/Downloads/Compressed/bank_transactions.csv.zip"),
    ]
    return next((path for path in candidates if path is not None and path.exists()), candidates[-1])


def prepare(source: Path, output_dir: Path, seed: int = 42) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {source}")
    with zipfile.ZipFile(source) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {source}, found {members}")
        member = members[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    raw_path = output_dir / "bank_transactions.csv"

    frame = pd.read_csv(source, compression="zip", usecols=SOURCE_COLUMNS, low_memory=False)
    frame["CustomerID"] = frame["CustomerID"].astype("string")
    frame.to_csv(raw_path, index=False)

    customers = np.array(sorted(frame["CustomerID"].dropna().unique().tolist()), dtype=object)
    rng = np.random.default_rng(seed)
    customers = customers[rng.permutation(len(customers))]
    test_count = int(round(len(customers) * 0.20))
    validation_count = int(round(len(customers) * 0.15))
    test_customers = set(customers[:test_count])
    validation_customers = set(customers[test_count : test_count + validation_count])

    split = frame["CustomerID"].map(
        lambda customer: "test"
        if customer in test_customers
        else "validation"
        if customer in validation_customers
        else "train"
    )
    rows = {}
    for name in ("train", "validation", "test"):
        part = frame.loc[split.eq(name)].copy()
        path = split_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        rows[name] = {
            "rows": int(len(part)),
            "customers": int(part["CustomerID"].nunique()),
            "path": str(path),
        }

    manifest = {
        "source_zip": str(source),
        "source_member": member,
        "seed": seed,
        "split_unit": "CustomerID",
        "raw_csv": str(raw_path),
        "rows": int(len(frame)),
        "customers": int(frame["CustomerID"].nunique()),
        "splits": rows,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Unzip and split bank_transactions.csv.zip")
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
