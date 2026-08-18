#!/usr/bin/env python3
# Read-only benchmark comparison entrypoint.
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "cryosparc_agent_remote"))

from benchmark_core import compare_benchmark_runs, write_compare_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--benchmark-root", default="reports/v24_benchmarks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_root) / args.benchmark_id
    if not benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory not found: {benchmark_dir}")
    report = compare_benchmark_runs(benchmark_dir)
    write_compare_outputs(benchmark_dir, report)
    print(f"Wrote comparison outputs under {benchmark_dir}")


if __name__ == "__main__":
    main()

