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
- [Dynamic agent routing](#dynamic-agent-routing)
- [Dynamic dataset paths and API](#http-api)
- [ML lifecycle](#ml-lifecycle)
- [Determinism and fallbacks](#determinism-and-fallbacks)
- [Episodic memory](#episodic-memory)
- [Testing](#testing)
- [Security and production checklist](#security-and-production-checklist)

## Capabilities

| Area | What it does |
| --- | --- |
| Customer segmentation | Assigns `priority`, `regular`, `dormant`, or `needs_review` using training-derived, explainable rules. |
| Data cleaning and filtering | Coerces timestamps/numerics, removes unusable rows and duplicates, audits dropped rows, and filters events to the configured lookback window. |
| Feature engineering | Builds balance, stability, activity, recency, transaction-size, and product-engagement features. |
| Feature selection | Uses leakage-safe mutual-information selection fitted on training data only. |
| Dimensionality reduction | Fits deterministic two-component PCA on selected training features for diagnostics and visual exploration. |
| Visualization | Saves segment distribution, PCA segment projection, and feature-by-segment distribution charts as PNG artifacts. |
| ML evaluation | Auto-tunes K-Means/GMM hyperparameters on train/validation only and checks K-Means stability with cross-validation. |
| Leakage prevention | Audits customer overlap, training-only thresholds/feature selection/PCA fitting, and confirms the test partition is never used for tuning. |
| Fit diagnostics | Compares train and validation silhouette scores and flags acceptable generalization, overfitting risk, or underfitting risk. |
| Candidate conversion | Ranks regular customers by deterministic distance to priority thresholds and proposes an action. |
| Agent workflow | Uses LangGraph for visible query planning, validation, engineering, evaluation, and recommendation stages. |
| Multi-agent trace | Labels specialist agents and tools for every workflow stage and exposes the trace as JSON/SSE. |
| Web UI | Local browser console at `/ui` with live progress, agent/tool cards, and complete JSON response. |
| Inter-agent protocol | Agent Card at `/.well-known/agent.json` and an A2A-friendly JSON-RPC endpoint at `/a2a`. |
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
        │ clean/filter → features → select → PCA → evaluate  │
        │ segment → visualize → recommend                    │
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
     │ CSV · JSON trace · PCA metadata · PNG diagnostics    │
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
git clone https://github.com/gaur-avvv/banking-segmentation-agent.git
cd banking-segmentation-agent
```

Recommended one-command setup:

```bash
# Linux, macOS, or WSL
bash scripts/setup.sh
```

```powershell
# Windows PowerShell
.\scripts\setup.ps1
```

```bat
:: Windows Command Prompt
scripts\setup.cmd
```

Each script creates/uses `.venv`, installs the development and Google ADK extras, asks for a dataset CSV/ZIP/folder path, saves it as `BANKING_DATA_PATH` in `.env`, creates `.env` from `.env.example` without overwriting existing settings, and prints the next commands. Press Enter at the dataset prompt to use the automatic safe demo fallback. On Linux/macOS/WSL, `uv` is used when available; otherwise standard Python `venv` and `pip` are used.

The setup scripts then offer a numbered start menu: terminal chat, local API/browser UI, Google ADK Web, one-shot segmentation, or exit. Use `--help` for script help:

```bash
bash scripts/setup.sh --help
```

```powershell
.\scripts\setup.ps1 --help
```

```bat
scripts\setup.cmd --help
```

After setup, the menu options do the following:

1. **Terminal chat** — enter multiple natural-language questions at `banking>`; use `/help`, `/json`, `/last`, `/trace on|off`, and `/quit`.
2. **Local API + browser UI** — starts `http://127.0.0.1:8000`; open `/ui` to select a dataset, enter a query, watch agent/tool events, and view PNG charts inline.
3. **Google ADK Web** — starts `http://127.0.0.1:8001`; select `adk_app` to watch root-agent transfers and specialist tool calls.
4. **One-shot query** — runs a default segmentation request and prints JSON/artifact paths.
5. **Exit** — leaves setup without starting a service.

You can skip the menu and run commands directly:

```bash
# Chat
uv run banking-agent chat --data-path /path/to/transactions.csv

# API and browser UI
uv run banking-agent api --data-path /path/to/transactions.csv --host 127.0.0.1 --port 8000

# Google ADK Web
uv run banking-agent adk-web --provider gemini --model gemma-4-26b-a4b-it \\
  --data-path /path/to/transactions.csv --port 8001

# One-shot query with deterministic local planning
uv run banking-agent run --provider none --data-path /path/to/transactions.csv \\
  --query "Which regular customers can be converted to priority?"
```

For WSL, translate a Windows file path to `/mnt/c/...`:

```bash
uv run banking-agent api \\
  --data-path /mnt/c/Users/Dell/Downloads/Compressed/bank_transactions.csv \\
  --port 8000
```

For PowerShell, keep the Windows path quoted:

```powershell
.\.venv\Scripts\banking-agent.exe api `
  --data-path "C:\Users\Dell\Downloads\Compressed\bank_transactions.csv" `
  --port 8000
```

Example questions:

```text
Segment customers into priority, regular, and dormant groups
Explain why priority customers were selected
What is the average transaction size for priority and regular customers?
Which regular customers can be converted to priority and what should be done?
Show the balance and transaction-frequency distributions by segment
```

Manual setup is still available:

```bash
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

### WSL quick start

From Ubuntu/WSL, `uv run` can launch the console even before a global `banking-agent` command exists:

```bash
python3 -m pip install --user uv
uv sync --extra adk
uv run banking-agent adk-web \\
  --provider gemini \\
  --model gemma-4-26b-a4b-it \\
  --data-path /mnt/c/Users/Dell/Downloads/Compressed/bank_transactions.csv \\
  --port 8001
```

Or install into an activated environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,adk]"
banking-agent adk-web --data-path /mnt/c/Users/Dell/Downloads/Compressed/bank_transactions.csv --port 8001
```

In WSL, Windows drives use `/mnt/c/...`; do not use `mnt\\c\\...`. If `banking-agent: command not found` appears, use `uv run banking-agent ...` or activate `.venv` first.

### OS-specific command reference

Linux/macOS:

```bash
source .venv/bin/activate
export GEMINI_API_KEY="your-key"
banking-agent setup
banking-agent chat
banking-agent api --host 127.0.0.1 --port 8000
banking-agent adk-web --provider gemini --model gemma-4-26b-a4b-it --port 8001
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:GEMINI_API_KEY = "your-key"
banking-agent setup
banking-agent adk-web --data-path "C:\Users\Dell\Downloads\Compressed\bank_transactions.csv" --port 8001
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
set GEMINI_API_KEY=your-key
banking-agent run --data-path "C:\Users\Dell\Downloads\Compressed\bank_transactions.csv" --query "Compare transaction sizes"
```

WSL:

```bash
source .venv/bin/activate
export GEMINI_API_KEY="your-key"
banking-agent adk-web --data-path /mnt/c/Users/Dell/Downloads/Compressed/bank_transactions.csv --port 8001
```

If the console script is unavailable, prefix the command with `uv run` (for example, `uv run banking-agent chat`).

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
GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-2.0-flash
```

Provider selection is automatic by default. Gemini is preferred when `GEMINI_API_KEY` exists, followed by OpenRouter, Groq, and OpenAI-compatible providers. A local Ollama model can be selected explicitly without a cloud key. You may select explicitly:

```dotenv
LLM_PROVIDER=auto          # auto, gemini, openai, openai-compatible, or none
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=           # optional OpenRouter/local compatible endpoint
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=gemma3:4b
```

Shell examples:

```bash
export GEMINI_API_KEY="your-gemini-key"
export GEMINI_MODEL="gemma-4-26b-a4b-it"
# or
export OPENAI_API_KEY="your-openai-key"
export OPENAI_MODEL="gpt-4o-mini"
# OpenRouter's free router (availability and limits change):
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENROUTER_MODEL="openrouter/free"
# Groq:
export GROQ_API_KEY="your-groq-key"
# Local Ollama:
ollama pull gemma3:4b
export OLLAMA_MODEL="gemma3:4b"
```

PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-gemini-key"
# or
$env:OPENAI_API_KEY = "your-openai-key"
# or
$env:OPENROUTER_API_KEY = "your-openrouter-key"
# or choose local Ollama
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "gemma3:4b"
```

Only the query is sent to the selected planner; banking records remain local. OpenRouter's `openrouter/free` router and `:free` model variants are subject to changing availability and rate limits; Groq free quotas also depend on the account and model. Ollama runs locally. If keys are absent, quota-limited, or produce invalid JSON, the agent logs the reason and uses its deterministic local router. API keys must never be committed or pasted into chat.

Gemma 4 remains the default model. For transient Gemini `503 UNAVAILABLE`, `429`, quota, or timeout errors, the query planner automatically tries the comma-separated `GEMINI_FALLBACK_MODELS` list in order (default: Gemini 2.5 Flash, then Gemini 2.0 Flash), and finally uses the deterministic local router. ADK Web keeps the configured Gemma model by default and uses ADK's HTTP retry policy; set `ADK_MODEL` explicitly if you want to run ADK Web on a Flash fallback model.

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

Every command accepts `--data-path` with an absolute or relative path to a CSV, ZIP, or dataset folder. The older `--data-dir` option remains a compatibility alias. The loader detects the canonical four-file format, the supplied `bank_transactions.csv` / ZIP, or a folder containing one transaction CSV/ZIP. Single transaction files support aliases such as `customer_id` / `CustomerID`, `timestamp` / `TransactionDate`, `amount` / `TransactionAmount (INR)`, and `balance` / `CustAccountBalance`. Customer, date/timestamp, and amount are required; missing balance history is safely marked for review. Paths are normalized across Linux, macOS, Windows, and WSL (for example, a `C:\\Users\\...` path works in WSL). If a requested path is missing, the agent automatically creates and uses the safe local demo dataset instead of failing with `FileNotFoundError`; the first `dataset_resolution` event records exactly what happened.

Generate synthetic demo data:

```bash
python scripts/generate_sample_data.py
banking-agent run --data-path data --query "Segment customers and find priority candidates"
```

Select a provider/model per command without editing `.env`:

```bash
banking-agent run --data-path data --provider openrouter --model openrouter/free \
  --query "Compare transaction sizes"
banking-agent chat --data-path data --provider ollama --model gemma3:4b
```

If you cloned the repository without a dataset, run the safe built-in demo:

```bash
banking-agent run --demo --query "Segment customers into priority, regular, and dormant groups"
banking-agent chat --demo
```

The demo creates a small non-sensitive dataset under `data/demo`. The public repository intentionally does not include the Kaggle banking CSV: its listing states “Data files © Original Authors,” and the file contains customer identifiers, demographics, balances, and transaction activity. Download it from the [Kaggle source listing](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation) only if your use and redistribution rights permit it, then pass its path with `--data-path`.

### First-run dataset setup

You can save a dataset path once for the clone:

```bash
banking-agent setup
```

Interactive terminal commands (`run`, `chat`, `api`, and `adk-web`) also ask for a CSV, ZIP, or dataset-folder path when no explicit path is supplied. Press Enter to use the automatic local fallback. Non-interactive scripts and API calls never block: they use `BANKING_DATA_PATH` or the safe `data/demo` dataset. The setup command writes only `BANKING_DATA_PATH` to the ignored local `.env` file.

Automatic downloads are intentionally not performed silently. Kaggle sources may require credentials and may restrict redistribution. If you have a permitted HTTPS/approved source, download it yourself (or place it in the clone) and pass the resulting path; this keeps credentials and banking records out of the agent.

### Interactive terminal chat

Use chat mode when you want to ask several questions in one terminal session and see how the agent reached each answer:

```bash
banking-agent chat --data-path data
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

Run against a file on Windows PowerShell:

```powershell
banking-agent run `
  --data-path 'C:\Users\Dell\Downloads\Compressed\bank_transactions.csv.zip' `
  --query 'Segment customers into priority, regular, and dormant groups'
```

Run against a folder on Linux/macOS/WSL:

```bash
banking-agent run \
  --data-path /absolute/path/to/dataset-folder \
  --query "Which regular customers can be converted to priority?"
```

The same command accepts the Windows path supplied to this project from WSL:

```bash
banking-agent run --provider none \\
  --data-path 'C:\\Users\\Dell\\Downloads\\Compressed\\bank_transactions.csv' \\
  --query "Segment customers into priority, regular, and dormant groups"
```

### HTTP API

Start the local API with a default dataset path:

```bash
banking-agent api --data-path data --host 127.0.0.1 --port 8000
```

Open the browser console:

```text
http://127.0.0.1:8000/ui
```

The console displays specialist-agent calls (`data_quality_agent`, `feature_agent`, `model_agent`, `governance_agent`, `recommendation_agent`, and `visualization_agent`), tool names, details, generated PNG visualizations inline, and the final JSON response. PNGs are served only from generated `artifacts/` directories through the local `/artifact` endpoint.

Check health and run a query:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"query":"Which regular customers can be converted to priority?"}'
```

The request may override the server's default path:

```json
{
  "data_path": "/absolute/path/to/transactions.csv",
  "query": "Compare average transaction size for priority and regular customers",
  "memory_consent": false,
  "provider": "ollama",
  "model": "gemma3:4b"
}
```

The API returns the same final report, event trace, leakage audit, fit diagnostics, recommendations, and artifact paths as the terminal CLI.

### Dynamic agent routing

The ADK root agent first calls `query_route_tool`. It normalizes the user query and activates the smallest useful route:

| Query pattern | Dynamic route |
| --- | --- |
| Segment, compare, average, recommend, convert | EDA → feature engineering → segmentation → explanation |
| Explain, why, audit, basis, review | Governance/explainability review |
| Unknown or ambiguous | Safe EDA/feature exploration fallback |

The Python API is also available for applications that want to inspect the route before creating an agent:

```python
from banking_agent.adk_multiagent import create_dynamic_agent, query_route_tool

query = "Why were priority customers selected?"
print(query_route_tool(query))
agent = create_dynamic_agent(query)
```

The route is deterministic, local, and cache-aware. Repeated unchanged tool calls reuse the local TTL cache. ADK Web streams transfers, tool calls, and structured results; it does not expose private chain-of-thought.

### A2A and Google ADK

The service exposes an Agent Card at `GET /.well-known/agent.json` and an A2A-friendly JSON-RPC façade at `POST /a2a`:

```bash
curl -X POST http://127.0.0.1:8000/a2a \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","params":{"query":"Compare transaction sizes","data_path":"data","provider":"none"}}'
```

Optional integrations:

```bash
python -m pip install -e ".[adk,a2a]"
```

The deterministic LangGraph workflow remains the safety/audit execution path. If `google-adk` is installed, `banking_agent.adk_adapter.create_adk_root_agent()` builds a real ADK root router with `SequentialAgent` and `LoopAgent` orchestration. Its specialist agents are `eda_agent`, `feature_engineering_agent`, `segmentation_agent`, `explainability_agent`, and `governance_explainability_agent`; their registered tools call the same local data, feature, segmentation, and explanation functions. If `a2a-sdk` is installed, it can replace the local façade with a full SDK server. The local UI and API remain usable without either optional package.

The repository includes `adk_app/agent.py` with the ADK-required `root_agent` symbol. Run it from the repository root:

```bash
python -m pip install -e ".[adk]"
export GOOGLE_API_KEY="your-gemini-key"
adk run adk_app
```

Start the ADK development web UI:

```bash
adk web --port 8001
```

The equivalent project command checks that Google ADK is installed and launches the same UI:

```bash
banking-agent adk-web \\
  --data-path '/absolute/path/to/bank_transactions.csv' \\
  --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001` and select `adk_app`. ADK Web is intended for development/debugging; the project's local console remains `banking-agent api`, which displays the deterministic specialist-agent/tool trace.

The local API console at `/ui` includes a dataset selector populated from `/datasets`, plus an optional custom path field. Leave the custom field empty to use the first discovered dataset; if no dataset exists, it uses `data/demo` automatically. This makes a fresh clone immediately runnable in the ADK web UI, terminal, or API without hard-coding `banking_data.csv`.

For ADK Web, `--data-path` is exported as `BANKING_DATA_PATH` before the child ADK process starts. Every specialist therefore receives the same dataset context. A missing path is resolved to the local demo and recorded in the tool result; the root router is instructed not to guess filenames or pause for a path that can be safely resolved.

ADK routing is query-aware: segmentation, comparison, and recommendation requests use the sequential analytics agents, while explanation/audit requests use the governance review loop. ADK tools use a local TTL cache keyed by dataset path and normalized query, so an unchanged request returns `cache_hit: true` instead of repeating EDA, feature extraction, or segmentation. The default cache TTL is 15 minutes and can be changed with `BANKING_AGENT_CACHE_TTL`.

ADK Web streams agent events by default. The launcher accepts `--provider` and `--model` and exports them to the ADK process:

```bash
banking-agent adk-web --provider gemini --model gemma-4-26b-a4b-it --port 8001
```

Gemini models are native to Google ADK. OpenRouter, Groq, OpenAI-compatible, and Ollama providers remain available through the local API/CLI planner; use their API keys and model names there. Generation is configured with low temperature and a 512-token cap to reduce quota usage. The system exposes concise tool summaries and decisions, not private chain-of-thought.

Run against a folder containing the supported ZIP:

```bash
banking-agent run \
  --data-path /path/to/dataset-folder \
  --query "Which regular customers can be converted to priority?"
```

Outputs are written beside the selected data directory:

```text
artifacts/customer_segments.csv
artifacts/run_report.json
artifacts/visualizations/segment_distribution.png
artifacts/visualizations/pca_segments.png
artifacts/visualizations/feature_distributions.png
```

`customer_segments.csv` includes the segment, reason, fallback level, assignment confidence, and engineered features. `run_report.json` includes the agent event trace, split sizes, thresholds, feature-selection metadata, PCA variance information, validation metrics, cross-validation results, and candidate recommendations. The PNG charts are diagnostic outputs; they are not used to make the deployable rule decision.

## ML lifecycle

1. **Validate:** inspect schema, row counts, duplicates, and missingness.
2. **Clean and filter:** coerce timestamps/numerics, remove invalid/duplicate event rows, and apply the lookback window.
3. **Engineer:** aggregate transaction and balance events to customer features.
4. **Split:** chronological split where time cohorts exist; deterministic customer split for a snapshot.
5. **Select:** fit mutual-information feature selection on training data only.
6. **Reduce:** fit two-component PCA on selected training features for stable diagnostics.
7. **Train:** evaluate K-Means and GMM on the held-out validation partition.
8. **Tune:** search deterministic K-Means/GMM grids and select by validation silhouette with a generalization-gap penalty.
9. **Fit diagnostics:** compare train/validation scores to detect overfitting or underfitting risk.
10. **Stability check:** run deterministic K-Fold evaluation on the training partition.
11. **Leakage audit:** verify disjoint customer IDs and that test data was not used to fit, select, tune, or threshold the workflow.
12. **Test:** report the rule baseline on untouched test data.
13. **Visualize:** save segment counts, PCA projections, and feature distribution charts.
14. **Serve:** persist the rule rationale and recommendation candidates.
15. **Monitor:** in production, monitor input drift, segment migration, candidate conversion, fairness, and model fallback rates.

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
| Train/validation silhouette gap exceeds the configured threshold | Logged `overfitting_risk`; the rule baseline remains the deployable decision layer. |
| Both train and validation silhouette scores are low | Logged `underfitting_risk`; review features, segment assumptions, and data volume. |
| Any customer appears in more than one partition | Leakage audit fails and the run report exposes overlap counts. |

All stochastic components use `random_state=42` by default. Rule thresholds come only from the training partition.

### What the terminal agent shows

The chat CLI prints a trace for each completed query. A typical run is:

```text
Query Planning → Data Validation → Data Cleaning Filtering → Feature Extraction
→ Feature Selection → Dimensionality Reduction → Hyperparameter Tuning
→ Fit Diagnostics → Leakage Audit → Model Evaluation → Recommendations → Visualization
```

The final summary includes the leakage-audit status, fit-diagnostic status for each tuned model, selected hyperparameters, segment counts, candidate actions, and artifact paths. Use `/json` in chat to inspect the complete report.

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
- [OpenRouter free models router](https://openrouter.ai/docs/guides/routing/routers/free-router)
- [OpenRouter model catalogue](https://openrouter.ai/docs/guides/overview/models)
- [Groq OpenAI compatibility](https://console.groq.com/docs/openai)
- [Ollama OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md)
