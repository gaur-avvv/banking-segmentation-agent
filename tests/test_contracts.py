import pandas as pd

from banking_agent.contracts import load_dataset


def test_loader_accepts_single_csv_with_common_aliases(tmp_path):
    source = tmp_path / "transactions.csv"
    pd.DataFrame(
        {
            "customer": ["a", "b"],
            "date": ["2025-01-01", "2025-01-02"],
            "amount": [10, 20],
            "account_balance": [100, 200],
        }
    ).to_csv(source, index=False)
    frames = load_dataset(source)
    assert set(frames) == {"customers", "balances", "transactions", "product_holdings"}
    assert len(frames["transactions"]) == 2
    assert frames["transactions"]["amount"].tolist() == [10, 20]


def test_loader_accepts_single_csv_inside_folder(tmp_path):
    source = tmp_path / "events.csv"
    pd.DataFrame({"CustomerID": ["a"], "TransactionDate": ["1/1/25"], "TransactionAmount (INR)": [5]}).to_csv(source, index=False)
    frames = load_dataset(tmp_path)
    assert frames["customers"]["customer_id"].tolist() == ["a"]
    assert frames["balances"]["balance"].isna().all()
