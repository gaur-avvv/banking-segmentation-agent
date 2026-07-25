from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED = {
    "customers": {"customer_id"},
    "balances": {"customer_id", "timestamp", "balance"},
    "transactions": {"customer_id", "timestamp", "amount"},
    "product_holdings": {"customer_id", "product_name", "status"},
}


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...], required: bool = True) -> str | None:
    columns = {_key(column): column for column in frame.columns}
    for alias in aliases:
        if _key(alias) in columns:
            return columns[_key(alias)]
    if required:
        raise ValueError(f"Could not find a column matching one of: {', '.join(aliases)}. Available columns: {list(frame.columns)}")
    return None


def _parse_timestamp_values(values: pd.Series) -> pd.Series:
    """Parse common banking date formats without pandas inference warnings."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns, UTC]")
    formats = (
        "%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z",
        "%d/%m/%y", "%d/%m/%Y", "%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d",
    )
    for date_format in formats:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format=date_format, errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format="mixed", errors="coerce", dayfirst=True, utc=True)
    return parsed


def load_dataset(data_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load a canonical folder, a transaction CSV/ZIP, or a folder containing one."""
    source = Path(data_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source}")
    if source.is_file():
        return _load_file(source)

    source_zip = source / "bank_transactions.csv.zip"
    if source_zip.exists():
        return load_bank_transactions_zip(source_zip)
    source_csv = source / "bank_transactions.csv"
    if source_csv.exists():
        return load_bank_transactions_csv(source_csv)

    canonical = {name: source / f"{name}.csv" for name in REQUIRED}
    if all(path.exists() for path in canonical.values()):
        frames = {name: pd.read_csv(path) for name, path in canonical.items()}
        for name, required in REQUIRED.items():
            missing = required - set(frames[name].columns)
            if missing:
                raise ValueError(f"{canonical[name]} missing columns: {sorted(missing)}")
        return frames

    candidates = sorted((*source.glob("*.csv"), *source.glob("*.zip")))
    if len(candidates) == 1:
        return _load_file(candidates[0])
    raise FileNotFoundError(
        f"Could not identify a dataset in {source}. Pass a CSV/ZIP file, a canonical folder "
        "(customers.csv, balances.csv, transactions.csv, product_holdings.csv), or a folder with one transaction CSV."
    )


def _load_file(path: Path) -> dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return load_bank_transactions_zip(path)
    if suffix == ".csv":
        return load_bank_transactions_csv(path)
    raise ValueError(f"Unsupported dataset file {path}; use .csv or .zip")


def load_bank_transactions_zip(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load a CSV inside a ZIP using flexible transaction-column aliases."""
    raw = pd.read_csv(path, compression="zip", low_memory=False)
    return _bank_transactions_frame_to_contract(raw)


def load_bank_transactions_csv(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load a transaction CSV using common banking column names."""
    raw = pd.read_csv(path, low_memory=False)
    return _bank_transactions_frame_to_contract(raw)


def _bank_transactions_frame_to_contract(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    customer_column = _find_column(raw, ("customer_id", "CustomerID", "customer", "client_id", "account_holder"))
    amount_column = _find_column(raw, ("amount", "transaction_amount", "TransactionAmount (INR)", "value", "debit", "credit"))
    timestamp_column = _find_column(raw, ("timestamp", "datetime", "transaction_datetime", "date", "transaction_date", "TransactionDate"), required=False)
    time_column = _find_column(raw, ("time", "transaction_time", "TransactionTime"), required=False)
    balance_column = _find_column(raw, ("balance", "account_balance", "CustAccountBalance", "current_balance"), required=False)
    if timestamp_column is None:
        raise ValueError("A timestamp/date column is required. Supported aliases include timestamp, datetime, date, and TransactionDate.")

    date_values = raw[timestamp_column].astype("string")
    if time_column is not None:
        time_values = raw[time_column].astype("string").str.replace(r"[^0-9]", "", regex=True).str.zfill(6)
        date_values = date_values + " " + time_values.str.slice(0, 2) + ":" + time_values.str.slice(2, 4) + ":" + time_values.str.slice(4, 6)
    timestamp = _parse_timestamp_values(date_values)
    customer = raw[customer_column].astype("string")
    customers = pd.DataFrame({"customer_id": customer}).dropna().drop_duplicates()
    common = pd.DataFrame({"customer_id": customer, "timestamp": timestamp})
    balances = common.assign(balance=pd.to_numeric(raw[balance_column], errors="coerce") if balance_column else float("nan"))
    transactions = common.assign(amount=pd.to_numeric(raw[amount_column], errors="coerce"))
    return {
        "customers": customers,
        "balances": balances,
        "transactions": transactions,
        "product_holdings": pd.DataFrame(columns=["customer_id", "product_name", "status"]),
    }


def data_quality_report(frames: dict[str, pd.DataFrame]) -> dict:
    report = {}
    for name, df in frames.items():
        report[name] = {
            "rows": len(df),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
        }
    return report
