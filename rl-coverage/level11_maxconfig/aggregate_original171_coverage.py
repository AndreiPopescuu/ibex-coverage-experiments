"""aggregate_original171_coverage.py — union toggle/branch/line coverage over
EXACTLY the original 171-test suite run (scaled_seed_counts()'s default
target_total=150), filtered by exact filename from cpu/coverage_suite_*.dat —
not a blind glob. This matters because the same directory can also contain
coverage_suite_*.dat from the separate extended-budget suite run (see
RL_VS_TESTLIST_SUITE_REPORT.md §8.2/8.3, up to riscv_pmp_suite_test_seed437
etc.), which would silently inflate a naive glob-based aggregate.

Usage:
    python3 aggregate_original171_coverage.py            # all 171, incl. riscv_csr_test
    python3 aggregate_original171_coverage.py --exclude riscv_csr_test
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPU = os.path.join(_HERE, "..", "..", "cpu")
sys.path.insert(0, os.path.join(_HERE, "..", "level5_real_rtl"))

import cov_parser  # noqa: E402
from testlist_l11 import scaled_seed_counts, PROFILES  # noqa: E402


def expected_filenames(exclude: set[str]) -> list[str]:
    counts = scaled_seed_counts()  # default target_total=150 -> the original 171-run split
    det_names = {p.name for p in PROFILES if p.deterministic}
    names = []
    for profile_name, n_seeds in counts.items():
        if profile_name in exclude:
            continue
        # deterministic profiles (riscv_csr_test) always map to exactly seed0,
        # regardless of n_seeds -- matches gen_testlist_corpora.py's own logic.
        seeds = [0] if profile_name in det_names else list(range(n_seeds))
        for seed in seeds:
            names.append(f"coverage_suite_{profile_name}_seed{seed}.dat")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[],
                     help="Profile name(s) to exclude entirely, e.g. riscv_csr_test")
    args = ap.parse_args()
    exclude = set(args.exclude)

    filenames = expected_filenames(exclude)
    print(f"Expected {len(filenames)} .dat files "
          f"(original-171 set{' minus ' + ', '.join(sorted(exclude)) if exclude else ''})")

    missing = [f for f in filenames if not os.path.exists(os.path.join(_CPU, f))]
    if missing:
        print(f"\n[!] {len(missing)} MISSING .dat files (not found in {_CPU}):")
        for f in missing:
            print(f"    {f}")
        print("\nRun replay_all_suite_corpora.sh (or the original suite run) to "
              "generate these before aggregating, otherwise the result below is "
              "an UNDER-count.\n")

    present = [f for f in filenames if f not in missing]

    union_hits = {cov_parser.KIND_TOGGLE: set(), cov_parser.KIND_BRANCH: set(),
                  cov_parser.KIND_LINE: set()}
    totals = {}

    for fname in present:
        summary = cov_parser.parse(os.path.join(_CPU, fname))
        for kind in union_hits:
            _, tot = summary.by_kind.get(kind, (0, 0))
            totals.setdefault(kind, tot)
            union_hits[kind] |= cov_parser.hit_set(summary, kind)

    print("=" * 70)
    print(f"AGGREGATE — original-171 set, {len(present)}/{len(filenames)} .dat files used"
          f"{' (excluding ' + ', '.join(sorted(exclude)) + ')' if exclude else ''}")
    print("=" * 70)
    for kind in (cov_parser.KIND_TOGGLE, cov_parser.KIND_BRANCH, cov_parser.KIND_LINE):
        tot = totals.get(kind, 0)
        cov = len(union_hits[kind])
        pct = 100.0 * cov / tot if tot else 0.0
        print(f"  {kind:8s}: {cov:>6}/{tot:<6}  {pct:6.2f}%")


if __name__ == "__main__":
    main()
