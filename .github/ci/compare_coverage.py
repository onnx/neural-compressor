"""Reject line or branch coverage drops greater than 0.05 percentage points."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def read_coverage(path):
    root = ET.parse(path).getroot()
    result = {}
    for metric in ("lines", "branches"):
        covered = int(root.attrib[f"{metric}-covered"])
        valid = int(root.attrib[f"{metric}-valid"])
        if not 0 <= covered <= valid or (metric == "lines" and valid == 0):
            raise ValueError(f"Invalid {metric} coverage in {path}: {covered}/{valid}")
        result[metric] = 100 * covered / valid if valid else 100.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    candidate = read_coverage(args.candidate)
    baseline = read_coverage(args.baseline)
    rows = ["| Coverage | Base | Candidate | Change |", "| --- | ---: | ---: | ---: |"]
    failed = []
    for metric in candidate:
        change = candidate[metric] - baseline[metric]
        rows.append(f"| {metric} | {baseline[metric]:.3f}% | {candidate[metric]:.3f}% | {change:+.3f} pp |")
        if change < -0.05 - 1e-9:
            failed.append(metric)
    report = "\n".join(rows) + "\n"
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(report)
    if failed:
        raise SystemExit(f"Coverage dropped by more than 0.05 percentage points: {', '.join(failed)}")


if __name__ == "__main__":
    main()
