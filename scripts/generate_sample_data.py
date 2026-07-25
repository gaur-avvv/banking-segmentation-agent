"""Generate non-sensitive synthetic banking data for a reproducible demo."""
import argparse
from pathlib import Path

from banking_agent.demo import ensure_demo_data


def main():
    parser = argparse.ArgumentParser(description="Generate non-sensitive synthetic banking data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parents[1] / "data")
    args = parser.parse_args()
    ensure_demo_data(args.output_dir)


if __name__ == "__main__":
    main()
