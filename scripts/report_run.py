from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a compact experiment report")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    lines = ["# Experiment summary", "", f"Identity: `{summary['experiment_identity']}`", "", "| Phase | Status | Metrics |", "|---|---|---|"]
    for name, result in summary["phases"].items():
        metrics = []
        for key in ("clean_ba", "backdoored_ba", "mean_asr", "poison_rate"):
            if key in result:
                metrics.append(f"{key}={result[key]}")
        lines.append(f"| {name} | complete | {', '.join(metrics) or '-'} |")
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
