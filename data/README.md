# Local dataset directory

This directory is intentionally kept out of Git. The transaction source contains
banking identifiers and financial activity, so it must remain on the local or
approved private storage system.

From the repository root, prepare the dataset with:

```bash
python scripts/prepare_bank_transactions.py \
  --source /path/to/bank_transactions.csv.zip \
  --output-dir data
```

On Windows PowerShell:

```powershell
python scripts/prepare_bank_transactions.py `
  --source 'C:\Users\Dell\Downloads\Compressed\bank_transactions.csv.zip' `
  --output-dir data
```

The command creates:

- `data/bank_transactions.csv` — unzipped local source;
- `data/splits/train.csv` — 65% of customers;
- `data/splits/validation.csv` — 15% of customers;
- `data/splits/test.csv` — 20% of customers; and
- `data/dataset_manifest.json` — row/customer counts and the split seed.

Splits are customer-disjoint and deterministic (`seed=42`) to prevent customer
history leakage. The agent can then run against the unzipped source:

```bash
banking-agent --data-dir data \
  --query "Segment customers into priority, regular, and dormant groups"
```
