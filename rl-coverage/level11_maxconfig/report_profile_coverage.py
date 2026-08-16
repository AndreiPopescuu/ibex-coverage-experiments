"""report_profile_coverage.py — union toggle/branch/line coverage across every
coverage_suite_<name>_seed*.dat for a single profile, printed per-seed and as a
union. Companion to aggregate_suite_coverage.py, which unions ALL profiles at once.

Usage (from level11_maxconfig/):
    python3 report_profile_coverage.py <profile_name>
"""
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPU = os.path.join(_HERE, "..", "..", "cpu")
sys.path.insert(0, os.path.join(_HERE, "..", "level5_real_rtl"))

import cov_parser  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 report_profile_coverage.py <profile_name>")
        sys.exit(1)
    name = sys.argv[1]

    paths = sorted(glob.glob(os.path.join(_CPU, f"coverage_suite_{name}_seed*.dat")))
    if not paths:
        print(f"ERROR: no coverage_suite_{name}_seed*.dat found in {_CPU}")
        sys.exit(1)

    union = {cov_parser.KIND_TOGGLE: set(), cov_parser.KIND_BRANCH: set(),
             cov_parser.KIND_LINE: set()}
    totals = {}
    for path in paths:
        s = cov_parser.parse(path)
        print(f"--- {os.path.basename(path)} ---")
        for kind in union:
            cov, tot = s.by_kind.get(kind, (0, 0))
            totals.setdefault(kind, tot)
            print(f"  {kind:7s}: {cov:>5}/{tot:<5} ({s.kind_pct(kind):.2f}%)")
            union[kind] |= cov_parser.hit_set(s, kind)

    print(f"\n=== {name}: union across {len(paths)} seed(s) ===")
    for kind in (cov_parser.KIND_TOGGLE, cov_parser.KIND_BRANCH, cov_parser.KIND_LINE):
        cov, tot = len(union[kind]), totals[kind]
        pct = 100.0 * cov / tot if tot else 0.0
        print(f"  {kind:7s}: {cov:>5}/{tot:<5} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
