#!/usr/bin/env python3
import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryosparc_agent_remote.openai_agents_runner import (
    config_from_args,
    parse_common_args,
    run_agents_closed_loop,
    smoke_api,
    write_json,
)


async def main_async(argv: list[str]) -> None:
    import argparse

    runs_parser = argparse.ArgumentParser(add_help=False)
    runs_parser.add_argument("--runs", type=int, default=3)
    runs_args, remaining = runs_parser.parse_known_args(argv)
    common_args = parse_common_args(remaining)
    root = Path(common_args.output_dir)
    summaries = []
    for index in range(1, runs_args.runs + 1):
        common_args.run_id = f"run_{index:03d}_{uuid.uuid4().hex[:8]}"
        run_dir = root / f"run_{index:03d}"
        config = config_from_args(common_args, output_dir=run_dir)
        if common_args.api_smoke_only:
            summary = await smoke_api(config)
            write_json(run_dir / "api_smoke.json", summary)
        else:
            summary = await run_agents_closed_loop(config)
        summaries.append(summary)
    write_json(root / "benchmark_summary.json", {"runs": summaries})
    print(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    asyncio.run(main_async(argv or sys.argv[1:]))


if __name__ == "__main__":
    main()
