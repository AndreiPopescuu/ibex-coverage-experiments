"""aggregate_suite_coverage.py — union toggle/branch/line coverage across every
coverage_suite_<name>.dat produced by cpu/run_testlist_suite.sh, the same way
lowRISC aggregates coverage across their 1530 individual UVM test runs: a
coverage point counts as "hit" for the suite if ANY single test hit it, not
just the last one run.

Usage (from cpu/, after running run_testlist_suite.sh):
    python3 ../rl-coverage/level11_maxconfig/aggregate_suite_coverage.py
"""
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPU = os.path.join(_HERE, "..", "..", "cpu")
sys.path.insert(0, os.path.join(_HERE, "..", "level5_real_rtl"))

import cov_parser  # noqa: E402

# lowRISC's own published opentitan report (see README.md / ibex.reports.lowrisc.org),
# generated with Cadence Xcelium + UVM + riscv-dv + Spike cosim on the same RTL commit
# this project vendored (8ed87e07) — NOT directly comparable, see caveats printed below.
LOWRISC_PUBLISHED = {
    cov_parser.KIND_TOGGLE: 90.0,
    cov_parser.KIND_BRANCH: 89.0,
    # lowRISC reports "statement" coverage; this project's Verilator build reports
    # "line" coverage instead — related but not the same metric, still shown for
    # rough reference.
    cov_parser.KIND_LINE: 94.9,
}


def main():
    dat_paths = sorted(glob.glob(os.path.join(_CPU, "coverage_suite_*.dat")))
    if not dat_paths:
        print(f"ERROR: no coverage_suite_*.dat found in {_CPU} — "
              f"run cpu/run_testlist_suite.sh first")
        sys.exit(1)

    union_hits = {cov_parser.KIND_TOGGLE: set(), cov_parser.KIND_BRANCH: set(),
                  cov_parser.KIND_LINE: set()}
    totals = {}
    per_test_new = []

    for path in dat_paths:
        name = os.path.basename(path)[len("coverage_suite_"):-len(".dat")]
        summary = cov_parser.parse(path)
        row = {"name": name}
        for kind in union_hits:
            cov, tot = summary.by_kind.get(kind, (0, 0))
            totals.setdefault(kind, tot)
            before = len(union_hits[kind])
            union_hits[kind] |= cov_parser.hit_set(summary, kind)
            row[kind] = (cov, tot, len(union_hits[kind]) - before)
        per_test_new.append(row)

    print("=" * 90)
    print(f"{'test':32s} {'toggle new':>12s} {'branch new':>12s} {'line new':>12s}")
    print("=" * 90)
    for row in per_test_new:
        t = row[cov_parser.KIND_TOGGLE][2]
        b = row[cov_parser.KIND_BRANCH][2]
        l = row[cov_parser.KIND_LINE][2]
        print(f"{row['name']:32s} {t:>12d} {b:>12d} {l:>12d}")
    print()

    print("=" * 90)
    print(f"AGGREGATE SUITE COVERAGE ({len(dat_paths)} test runs)")
    print("=" * 90)
    for kind in (cov_parser.KIND_TOGGLE, cov_parser.KIND_BRANCH, cov_parser.KIND_LINE):
        tot = totals.get(kind, 0)
        cov = len(union_hits[kind])
        pct = 100.0 * cov / tot if tot else 0.0
        published = LOWRISC_PUBLISHED[kind]
        print(f"  {kind:8s}: {cov:>6}/{tot:<6}  {pct:6.2f}%   "
              f"(lowRISC published: {published:.1f}%, delta {pct - published:+.2f}pp)")
    print()

    print("-" * 90)
    print("NOT a like-for-like comparison — same caveats as UPSTREAM_COMPARISON_SUMMARY.txt:")
    print("  - Verilator (open-source) vs lowRISC's Cadence Xcelium (commercial); different")
    print("    toggle-coverage instrumentation algorithms on identical RTL/parameters.")
    print("  - No Spike co-simulation here — instructions execute, but nothing checks the")
    print("    resulting architectural state is CORRECT, only that logic toggled.")
    print("  - No debug module / interrupt agent in this testbench — 30 of lowRISC's 57")
    print("    testlist.yaml tests are entirely skipped (see testlist_l11.py SKIPPED),")
    print("    and several more are partial/best-effort proxies (see PROFILES descriptions).")
    print("  - This is 16 test runs, not lowRISC's 1530 (49 categories x randomized seeds).")
    print("-" * 90)


if __name__ == "__main__":
    main()
