from __future__ import annotations

import argparse
from pathlib import Path

from run_optimization import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cooling retrofit optimization.")
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("config.json"),
        help="Path to optimization config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.config)
    print(f"Saved {len(result['all_evaluations'])} evaluations.")
    print(f"Saved {len(result['pareto'])} Pareto solutions.")
    print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
