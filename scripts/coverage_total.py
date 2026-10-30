#!/usr/bin/env python3
"""The total line coverage of a coverage.xml, and whether it fell below main's.

Usage:
  scripts/coverage_total.py total coverage.xml
  scripts/coverage_total.py compare coverage.xml --baseline 81.15 [--tolerance 0.1]

`total` prints the percentage, two decimals: what main stores beside its badge
(coverage-total.txt on the coverage-badge branch). `compare` exits 1 when the
pull request's total is lower than that by more than the tolerance, which is
there because the total moves by a few hundredths with no change to the code
(a test that times out one branch, a retry). The verdict, either way, goes to
stdout as one markdown line for the job summary.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def total(report: Path) -> float:
    """Covered lines over all lines, in percent, as coverage.py's xml says it."""
    rate = ET.parse(report).getroot().get("line-rate")
    if rate is None:
        raise ValueError(f"{report} has no line-rate: not a coverage.py xml report")
    return round(float(rate) * 100, 2)


def verdict(current: float, baseline: float, tolerance: float) -> tuple[bool, str]:
    delta = round(current - baseline, 2)
    sign = "+" if delta >= 0 else ""
    line = f"Total coverage {current:.2f}% ({sign}{delta:.2f} against main's {baseline:.2f}%)"
    if delta < -tolerance:
        return False, f"❌ {line}: it may not drop by more than {tolerance:.2f}."
    return True, f"✅ {line}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("total").add_argument("report", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("report", type=Path)
    compare.add_argument("--baseline", type=float, required=True)
    compare.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args(argv)

    current = total(args.report)
    if args.command == "total":
        print(f"{current:.2f}")
        return 0
    ok, line = verdict(current, args.baseline, args.tolerance)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
