from __future__ import annotations

import argparse
import json
from .agent import run_agent


def main():
    parser = argparse.ArgumentParser(description="Run the banking segmentation agent")
    parser.add_argument("--data-dir", default="data", help="Directory containing the four required CSV files")
    parser.add_argument("--query", default="Segment customers and find priority conversion candidates")
    parser.add_argument("--user-id", help="Optional analytics-user ID for consented episodic memory")
    parser.add_argument("--memory-db", help="SQLite path for local episodic memory")
    parser.add_argument("--memory-consent", action="store_true", help="Record this interaction only when explicit consent exists")
    args = parser.parse_args()
    print(json.dumps(run_agent(args.data_dir, args.query, args.user_id, args.memory_db, args.memory_consent), indent=2, default=str))


if __name__ == "__main__":
    main()
