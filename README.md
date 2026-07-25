# Banking Segmentation & Personalization Agent

An auditable, local-first retail-banking analytics agent. It turns transaction history into explainable customer segments, evaluates exploratory clustering safely, identifies conversion candidates, and can retain consented analytics-session memories locally.

> **Status:** local development reference implementation. It does not make credit, lending, eligibility, or campaign-delivery decisions.

## Contents

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Requirements and installation](#requirements-and-installation)
- [Configuration](#configuration)
- [Data formats](#data-formats)
- [Run the agent](#run-the-agent)
- [ML lifecycle](#ml-lifecycle)
- [Determinism and fallbacks](#determinism-and-fallbacks)
- [Episodic memory](#episodic-memory)
- [Testing](#testing)
- [Security and production checklist](#security-and-production-checklist)

## Capabilities

| Area | What it does |
| --- | --- |
| Customer segmentation | Assigns `priority`, `regular`, `dormant`, or `needs_review` using training-derived, explainable rules. |
| Feature engineering | Builds balance, stability, activity, recency, transaction-size, and product-engagement features. |
| ML evaluation | Evaluates K-Means and GMM on held-out validation data and checks K-Means stability with cross-validation. |
| Candidate conversion | Ranks regular customers by deterministic distance to priority thresholds and proposes an action. |
| Agent workflow | Uses LangGraph for visible query planning, validation, engineering, evaluation, and recommendation stages. |
| LLM planning | Uses hosted Gemma 4 when configured; otherwise a deterministic keyword router takes over. |
| Episodic memory | Provides consent-gated SQLite interaction memory, deterministic retrieval, profile rebuilding, and deletion. |
| Auditability | Persists segments and a JSON event trace with data checks, model results, fallbacks, and decisions. |

## Architecture

```text
                     ┌──────────────────────────┐
                     │ User query / CLI request │
                     └────────────┬─────────────┘
                                  │
                 ┌────────────────▼────────────────┐
                 │ Query planner                    │
                 │ Gemma 4 → deterministic fallback │
                 └────────────────┬────────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │ LangGraph workflow                                 │
        │ validate → features → evaluate/segment → recommend │
        └───────┬───────────────────────┬────────────────────┘
                │                       │
     ┌──────────▼──────────┐  ┌─────────▼─────────────┐
     │ Customer feature set │  │ Optional memory store │
     │ balances, activity,  │  │ SQLite, consented,    │
     │ recency, amounts     │  │ customer-scoped       │
     └──────────┬──────────┘  └───────────────────────┘
                │
     ┌──────────▼──────────────────────────────────────────┐
     │ Rule baseline (deployable) + K-Means/GMM evaluation  │
     └──────────┬──────────────────────────────────────────┘
                │
     ┌──────────▼──────────────────────────────────────────┐
     │ CSV segments · JSON run report · execution events    │
     └─────────────────────────────────────────────────────┘
```

### Why this design is different

- **Rules are the production decision layer.** Clustering is evaluated for discovery and monitoring, but cannot silently replace policy.
- **No leakage by design.** Thresholds, imputation, scaling, and feature selection originate in training data—not validation or final-test data.
- **Graceful degradation.** Missing Gemini quota, malformed model plans, undersized data, or failed exploratory models result in logged deterministic fallbacks.
- **Memory without external embeddings.** The initial memory adapter keeps consented interactions local and uses stable keyword/metadata retrieval.
- **Privacy-first defaults.** Memory is off by default, requires explicit consent, and hashes rather than storing raw agent query text.

## Requirements and installation

### Supported systems

- Linux (Ubuntu/Debian, Fedora, etc.)
- macOS
- Windows 10/11 through PowerShell or Windows Subsystem for Linux (WSL)

Install Python **3.10 or newer** and Git. For datasets around one million rows, use at least 8 GB RAM; 16 GB is preferable.

### Clone and install

```bash
git clone https://github.com/YOUR-ACCOUNT/banking-segmentation-agent.git
cd banking-segmentation-agent
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package and test tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If your OS-managed Python cannot create environments, install its `venv` package or use [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"
```

## Configuration

Copy the template; `.env` is ignored by Git and must never be committed.

```bash
cp .env.example .env             # Linux / macOS / WSL
Copy-Item .env.example .env      # Windows PowerShell
```

Set `GEMINI_API_KEY` locally. The default configured planner is `gemma-4-26b-a4b-it`, a hosted Gemma 4 model. Google documents Gemma 4 hosted model IDs and usage [here](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api). The model only receives the natural-language query; banking records remain local.

```dotenv
GEMINI_API_KEY=replace-with-your-new-key
GEMINI_MODEL=gemma-4-26b-a4b-it
```

Gemma provides planning assistance; LangGraph executes the deterministic workflow. If the key is absent, quota-limited, or produces invalid JSON, the agent logs the reason and uses its local router.

## Data formats

### Canonical multi-file format

| File | Required columns |
| --- | --- |
| `customers.csv` | `customer_id` |
| `balances.csv` | `customer_id`, `timestamp`, `balance` |
| `transactions.csv` | `customer_id`, `timestamp`, `amount` |
| `product_holdings.csv` | `customer_id`, `product_name`, `status` |

### Supported single-ZIP format

Place `bank_transactions.csv.zip` in the data directory. The loader automatically maps these columns:

```text
CustomerID
CustAccountBalance
TransactionDate
TransactionTime
TransactionAmount (INR)
```

Only fields used by the pipeline are loaded. Demographics and other source columns are deliberately not retained by this adapter.

### Prepare the supplied ZIP locally

The repository does not commit the banking dataset. The source includes customer identifiers, balances, transaction amounts, DOB, gender, and location, so it must remain in approved local/private storage. A clone can recreate the working dataset and deterministic customer-disjoint splits with the included preparation script:

```bash
python scripts/prepare_bank_transactions.py \
  --source /path/to/bank_transactions.csv.zip \
  --output-dir data
```

Windows PowerShell:

```powershell
python scripts/prepare_bank_transactions.py `
  --source 'C:\Users\Dell\Downloads\Compressed\bank_transactions.csv.zip' `
  --output-dir data
```

This creates `data/bank_transactions.csv`, `data/splits/train.csv`, `data/splits/validation.csv`, `data/splits/test.csv`, and `data/dataset_manifest.json`. The split unit is `CustomerID` with seed `42`, so no customer's transactions cross train, validation, and test. The agent reads the unzipped `data/bank_transactions.csv` automatically, while its ML lifecycle also derives customer-level train/validation/test partitions and reports them in `artifacts/run_report.json`.

The generated files are ignored by Git. See [data/README.md](data/README.md) for the local workflow.

## Run the agent

Generate synthetic demo data:

```bash
python scripts/generate_sample_data.py
banking-agent --data-dir data --query "Segment customers and find priority candidates"
```

### Interactive terminal chat

Use chat mode when you want to ask several questions in one terminal session and see how the agent reached each answer:

```bash
banking-agent chat --data-dir data
```

At the `banking>` prompt, enter questions such as:

```text
Segment customers into priority, regular, and dormant groups
What is the average transaction size for priority and regular customers?
Which regular customers can be converted to priority?
```

After each query, the CLI prints the planned intent, segment counts, candidate actions, artifact paths, and the LangGraph event trace (`planning → validation → features → evaluation → recommendations`). The trace makes fallback behavior and the deterministic workflow visible.

Built-in chat commands:

```text
/help          Show available commands
/trace on|off  Toggle the event trace
/last          Reprint the last summary
/json          Print the complete last result as JSON
/quit          Exit the session
```

To use consented local memory in chat, add `--user-id`, `--memory-db`, and `--memory-consent`. The one-shot command remains available for scripts and automation.

Run against a folder containing the supported ZIP:

```bash
banking-agent \
  --data-dir /path/to/dataset-folder \
  --query "Which regular customers can be converted to priority?"
```

Outputs are written beside the selected data directory:

```text
artifacts/customer_segments.csv
artifacts/run_report.json
```

`customer_segments.csv` includes the segment, reason, fallback level, assignment confidence, and engineered features. `run_report.json` includes the agent event trace, split sizes, thresholds, validation metrics, cross-validation results, and candidate recommendations.

## ML lifecycle

1. **Validate:** inspect schema, row counts, duplicates, and missingness.
2. **Engineer:** aggregate transaction and balance events to customer features.
3. **Split:** chronological split where time cohorts exist; deterministic customer split for a snapshot.
4. **Train:** fit preprocessing and feature selection only on training data.
5. **Validate:** evaluate K-Means and GMM; calculate silhouette and Davies–Bouldin metrics.
6. **Stability check:** run deterministic K-Fold evaluation on the training partition.
7. **Test:** report the rule baseline on untouched test data.
8. **Serve:** persist the rule rationale and recommendation candidates.
9. **Monitor:** in production, monitor input drift, segment migration, candidate conversion, fairness, and model fallback rates.

### Engineered features

```text
avg_balance_90d                min_balance_90d
balance_stability_90d          transaction_count_90d
transaction_frequency_monthly  avg_transaction_amount
median_transaction_amount      recency_days
active_product_count
```

## Determinism and fallbacks

| Situation | Behavior |
| --- | --- |
| Gemini missing, unavailable, quota-limited, or malformed | Deterministic keyword router selects an intent and flags ambiguity. |
| Missing balance history | Customer is assigned `needs_review`, never inferred as priority. |
| Clear priority/dormant conditions | `L1`, high-confidence rule assignment. |
| Active but below priority thresholds | `L2`, medium-confidence regular assignment. |
| Missing-history exception | `L3`, low-confidence human review. |
| K-Means/GMM failure or insufficient validation rows | Logged `fallback_to_rules`; rule segmentation still completes. |

All stochastic components use `random_state=42` by default. Rule thresholds come only from the training partition.

## Episodic memory

Memory is optional and disabled unless all three arguments below are supplied:

```bash
banking-agent \
  --data-dir data \
  --user-id analyst-123 \
  --memory-db artifacts/memory.db \
  --memory-consent
```

The SQLite implementation provides:

- explicit-consent enforcement;
- customer-scoped deterministic keyword + recency retrieval;
- rebuildable preference/product-interest profile;
- de-identified audit events;
- `forget_user(user_id)` deletion of source memories; and
- a `MemoryStore`/`PayloadCipher` interface for a later PostgreSQL + pgvector/Qdrant or KMS/SQLCipher implementation.

`PlaintextCipher` is development-only. Use SQLCipher or a KMS-backed cipher before storing sensitive production data.

## Testing

```bash
pytest -q
```

The suite covers feature handling, deterministic routing, fallback levels, memory consent, retrieval isolation, deletion, and ineligible recommendation protection.

## Security and production checklist

- Rotate any API key that was exposed in chat, source code, logs, or screenshots.
- Keep `.env`, SQLite memory databases, data files, and artifacts outside Git.
- Encrypt data at rest and in transit; use a managed key service in production.
- Apply RBAC, least privilege, audit logging, data retention policies, and deletion workflows.
- Do not send raw banking records or PII to hosted LLMs without documented consent, data-processing approval, and legal/compliance review.
- Add fairness testing, model registry/versioning, approval gates, and campaign contact controls before operational use.
- Treat recommendations as decision support until a compliance-approved policy is in place.

## Model references

- [Gemma 4 on the Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
- [Gemma model overview](https://ai.google.dev/gemma/docs)
- [Gemini API model catalogue](https://ai.google.dev/gemini-api/docs/models)
- [Gemini API pricing and free-tier details](https://ai.google.dev/gemini-api/docs/pricing)
