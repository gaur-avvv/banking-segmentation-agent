from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {
    "customers": {"customer_id"},
    "balances": {"customer_id", "timestamp", "balance"},
    "transactions": {"customer_id", "timestamp", "amount"},
    "product_holdings": {"customer_id", "product_name", "status"},
}


def load_dataset(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    source_zip = root / "bank_transactions.csv.zip"
    if source_zip.exists():
        return load_bank_transactions_zip(source_zip)
    source_csv = root / "bank_transactions.csv"
    if source_csv.exists():
        return load_bank_transactions_csv(source_csv)
    frames: dict[str, pd.DataFrame] = {}
    for name, required in REQUIRED.items():
        path = root / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Required data file missing: {path}")
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv missing columns: {sorted(missing)}")
        frames[name] = frame
    return frames


def load_bank_transactions_zip(path: str | Path) -> dict[str, pd.DataFrame]:
    """Adapt the supplied Kaggle-style transaction ZIP into the canonical input contract."""
    # Keep only fields used by this pipeline. This avoids retaining unused PII and
    # is important for the supplied million-row source on local machines.
    needed = ["CustomerID", "CustAccountBalance", "TransactionDate", "TransactionTime", "TransactionAmount (INR)"]
    raw = pd.read_csv(path, compression="zip", usecols=needed, low_memory=False)
    return _bank_transactions_frame_to_contract(raw)


def load_bank_transactions_csv(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the unzipped local copy produced by prepare_bank_transactions.py."""
    needed = ["CustomerID", "CustAccountBalance", "TransactionDate", "TransactionTime", "TransactionAmount (INR)"]
    raw = pd.read_csv(path, usecols=needed, low_memory=False)
    return _bank_transactions_frame_to_contract(raw)


def _bank_transactions_frame_to_contract(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"CustomerID", "CustAccountBalance", "TransactionDate", "TransactionTime", "TransactionAmount (INR)"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"bank_transactions.csv.zip missing columns: {sorted(missing)}")
    date = pd.to_datetime(raw["TransactionDate"], format="%d/%m/%y", errors="coerce")
    time = raw["TransactionTime"].astype("string").str.zfill(6)
    timestamp = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + time.str.slice(0, 2) + ":" + time.str.slice(2, 4) + ":" + time.str.slice(4, 6), errors="coerce", utc=True)
    customers = raw[["CustomerID"]].rename(columns={"CustomerID": "customer_id"}).drop_duplicates()
    common = pd.DataFrame({"customer_id": raw["CustomerID"], "timestamp": timestamp})
    balances = common.assign(balance=pd.to_numeric(raw["CustAccountBalance"], errors="coerce"))
    transactions = common.assign(amount=pd.to_numeric(raw["TransactionAmount (INR)"], errors="coerce"))
    return {"customers": customers, "balances": balances, "transactions": transactions,
            "product_holdings": pd.DataFrame(columns=["customer_id", "product_name", "status"])}


def data_quality_report(frames: dict[str, pd.DataFrame]) -> dict:
    report = {}
    for name, df in frames.items():
        report[name] = {
            "rows": len(df),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
        }
    return report
