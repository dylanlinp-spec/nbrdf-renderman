"""
compare_with_published.py -- check your reproduced tables against the
published ones.

For every CSV in results/tables/ (the original results) it looks for a file
with the same name in results/reproduced/ (your run) and reports the
largest difference in any shared numeric column.

How to read the verdicts:
  MATCH      every value agrees to 6 significant figures.
  CLOSE      largest difference is small (<= 0.005 absolute). Expected for
             anything that went through the renderer on a different
             computer, and for the theta_o = 0 row of the albedo tables
             (see doc/known_issues.md).
  DIFFERENT  larger differences. Read doc/troubleshooting.md.
  NOT RUN    the reproduced file does not exist (that step was skipped).
Timing results depend on your hardware, so they are printed, not judged.

Usage:
    python src/evaluation/compare_with_published.py
    python src/evaluation/compare_with_published.py --published results/tables --reproduced results/reproduced
"""
import argparse
import csv
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLOSE_TOLERANCE = 0.005


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def compare_tables(published_rows, reproduced_rows):
    """Largest absolute and relative difference over shared numeric columns.
    Rows are matched by position; returns None if the shapes differ."""
    if len(published_rows) != len(reproduced_rows) or not published_rows:
        return None
    shared = [c for c in published_rows[0] if c in reproduced_rows[0]]
    max_abs, max_rel = 0.0, 0.0
    for pub, rep in zip(published_rows, reproduced_rows):
        for column in shared:
            a, b = as_float(pub[column]), as_float(rep[column])
            if a is None or b is None:
                if pub[column] != rep[column]:
                    return None  # a label column disagrees: rows are not aligned
                continue
            diff = abs(a - b)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, diff / max(abs(a), 1e-12))
    return max_abs, max_rel


def verdict(max_abs, max_rel):
    if max_rel <= 1e-6 or max_abs <= 1e-9:
        return "MATCH"
    if max_abs <= CLOSE_TOLERANCE:
        return "CLOSE"
    return "DIFFERENT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", default=os.path.join(REPO_ROOT, "results", "tables"))
    ap.add_argument("--reproduced", default=os.path.join(REPO_ROOT, "results", "reproduced"))
    args = ap.parse_args()

    print(f"{'table':<40} {'verdict':<10} {'max |diff|':>12}")
    print("-" * 64)
    for name in sorted(os.listdir(args.published)):
        if not name.endswith(".csv"):
            continue
        published_path = os.path.join(args.published, name)
        reproduced_path = os.path.join(args.reproduced, name)
        if name == "timing.csv":
            continue
        if not os.path.exists(reproduced_path):
            print(f"{name:<40} {'NOT RUN':<10}")
            continue
        result = compare_tables(read_rows(published_path), read_rows(reproduced_path))
        if result is None:
            print(f"{name:<40} {'DIFFERENT':<10} {'(rows/labels differ)':>12}")
            continue
        max_abs, max_rel = result
        print(f"{name:<40} {verdict(max_abs, max_rel):<10} {max_abs:12.3g}")

    timing_pub = os.path.join(args.published, "timing.csv")
    timing_rep = os.path.join(args.reproduced, "timing.csv")
    print("\nWhole-render timing (hardware-dependent, not judged):")
    for label, path in (("published", timing_pub), ("yours", timing_rep)):
        if os.path.exists(path):
            rows = {r["label"]: r["median_s"] for r in read_rows(path)}
            print(f"  {label:<10} " + "   ".join(f"{k} {v} s" for k, v in rows.items()))
        else:
            print(f"  {label:<10} (not run)")


if __name__ == "__main__":
    main()
