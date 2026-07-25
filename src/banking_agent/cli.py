from __future__ import annotations

import argparse
import json
from typing import Any

from .agent import run_agent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the banking segmentation agent or open its terminal chat"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("chat",),
        help="Use 'chat' for an interactive terminal session; omit for one-shot JSON output",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the supported CSV files or bank_transactions.csv.zip",
    )
    parser.add_argument("--query", default="Segment customers and find priority conversion candidates")
    parser.add_argument("--user-id", help="Optional analytics-user ID for consented episodic memory")
    parser.add_argument("--memory-db", help="SQLite path for local episodic memory")
    parser.add_argument(
        "--memory-consent",
        action="store_true",
        help="Record interactions only when explicit consent exists",
    )
    parser.add_argument(
        "--trace",
        choices=("on", "off"),
        default="on",
        help="Show the agent event trace in chat mode (default: on)",
    )
    return parser


def _print_help() -> None:
    print(
        "Commands:\n"
        "  /help          Show this help\n"
        "  /trace on|off  Toggle the workflow event trace\n"
        "  /json          Print the last result as JSON\n"
        "  /last          Reprint the last result summary\n"
        "  /quit          Exit the chat\n\n"
        "Enter a natural-language analytics question to run the agent."
    )


def _trace_plan(events: list[dict[str, Any]]) -> str | None:
    """Extract the planner intent without exposing raw provider response details."""
    for item in events:
        if item.get("step") != "query_planning":
            continue
        try:
            plan = json.loads(item.get("detail", "{}"))
        except (TypeError, json.JSONDecodeError):
            return None
        return str(plan.get("intent", "unknown"))
    return None


def _print_trace(events: list[dict[str, Any]]) -> None:
    print("\nAgent trace")
    print("-----------")
    for index, item in enumerate(events, start=1):
        step = str(item.get("step", "unknown")).replace("_", " ").title()
        detail = str(item.get("detail", "")).replace("\n", " ")
        print(f"{index:>2}. {step}: {detail}")


def _print_summary(result: dict[str, Any], show_trace: bool = True) -> None:
    report = result.get("report", {})
    print("\nFinal output")
    print("------------")
    intent = _trace_plan(result.get("events", []))
    if intent:
        print(f"Intent: {intent}")
    counts = report.get("segment_counts", {})
    if counts:
        print("Segments: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    leakage = report.get("leakage_prevention", {})
    if leakage:
        print(f"Leakage audit: {leakage.get('status', 'unknown')}")
    fit_checks = report.get("fit_diagnostics", {})
    if fit_checks:
        print("Fit diagnostics: " + ", ".join(f"{name}={detail.get('status', 'unknown')}" for name, detail in fit_checks.items()))
    tuned = report.get("unsupervised_validation", {})
    tuned_models = [name for name, detail in tuned.items() if isinstance(detail, dict) and detail.get("status") == "tuned"]
    if tuned_models:
        print("Auto-tuned models: " + ", ".join(tuned_models))
    candidates = report.get("top_priority_candidates", [])
    if candidates:
        print(f"Conversion candidates: {len(candidates)} shown")
        for candidate in candidates[:5]:
            print(
                "  - "
                f"{candidate.get('customer_id')}: "
                f"{candidate.get('recommended_action', 'review')}"
            )
    artifacts = result.get("artifacts", [])
    if artifacts:
        print("Artifacts:")
        for artifact in artifacts:
            print(f"  - {artifact}")
    if show_trace:
        _print_trace(result.get("events", []))


def _chat(args: argparse.Namespace) -> None:
    print("Banking Segmentation Agent — terminal chat")
    print("Type /help for commands or /quit to exit.")
    show_trace = args.trace == "on"
    last_result: dict[str, Any] | None = None

    while True:
        try:
            query = input("\nbanking> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not query:
            continue
        if query.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            return
        if query.lower() == "/help":
            _print_help()
            continue
        if query.lower().startswith("/trace"):
            value = query.split(maxsplit=1)[1].lower() if len(query.split()) > 1 else ""
            if value in {"on", "off"}:
                show_trace = value == "on"
                print(f"Trace {'enabled' if show_trace else 'disabled'}.")
            else:
                print("Usage: /trace on|off")
            continue
        if query.lower() == "/json":
            if last_result is None:
                print("No completed query yet.")
            else:
                print(json.dumps(last_result, indent=2, default=str))
            continue
        if query.lower() == "/last":
            if last_result is None:
                print("No completed query yet.")
            else:
                _print_summary(last_result, show_trace=show_trace)
            continue

        print("\nRunning the agent: planning → validation → features → evaluation → recommendations")
        try:
            last_result = run_agent(
                args.data_dir,
                query,
                args.user_id,
                args.memory_db,
                args.memory_consent,
            )
            _print_summary(last_result, show_trace=show_trace)
        except Exception as exc:  # Keep the REPL alive for the next query.
            print(f"Agent error: {exc}")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "chat":
        _chat(args)
        return
    print(
        json.dumps(
            run_agent(args.data_dir, args.query, args.user_id, args.memory_db, args.memory_consent),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
