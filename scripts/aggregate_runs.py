from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate completed experiment summaries")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("aggregate.json"))
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.root.glob("**/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    values: dict[str, list[float]] = {}
    for summary in summaries:
        for result in summary.get("phases", {}).values():
            for key in ("clean_ba", "backdoored_ba", "mean_asr", "poison_rate"):
                if isinstance(result.get(key), (int, float)):
                    values.setdefault(key, []).append(float(result[key]))
    aggregate = {key: {"count": len(items), "mean": statistics.fmean(items), "stdev": statistics.stdev(items) if len(items) > 1 else 0.0} for key, items in values.items()}
    args.output.write_text(json.dumps({"runs": len(summaries), "metrics": aggregate}, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
