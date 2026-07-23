#!/bin/bash
# run_multiseed_experiment.sh — empirical check for "is there room for more coverage
# by running more iterations per test, like lowRISC's testlist.yaml `iterations` counts
# (e.g. riscv_pmp_full_random_test: 600)?"
#
# The first suite run (2026-07-22) used 1 seed/profile and got 70.74% toggle / 67.19%
# branch / 50.25% line, vs lowRISC's published 90.0%/89.0%/94.9%. Several profiles
# contributed 0 NEW toggle points on top of what ran before them (see
# TESTLIST_SUITE_SUMMARY.txt section 4) — a sign of diminishing returns from the
# current instruction-mix strategies, not proof that more seeds can't help.
#
# This script regenerates every non-deterministic profile with 3 seeds instead of 1
# (--clean wipes old corpus_suite_*.json first), reruns the full suite, and
# re-aggregates — so you can directly compare the new % against the numbers above and
# see how much headroom more iterations actually buy before reaching directly for
# harder levers (bigger n_actions per profile, or implementing more of the SKIPPED
# categories in testlist_l11.py).
#
# Expect ~45 Verilator/cocotb runs total (~90-150s each => roughly 60-90 min).
#
# Usage (from anywhere, this cds internally):
#   bash rl-coverage/level11_maxconfig/run_multiseed_experiment.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${IBEX_VENV:-$HOME/ibex_env}/bin/activate"

cd "$REPO/rl-coverage/level11_maxconfig"
echo "=== Regenerating corpora: 3 seeds/profile (deterministic profiles stay at 1) ==="
python3 gen_testlist_corpora.py --clean --seed 0 --n-seeds 3

echo ""
echo "=== Clearing old coverage_suite_*.dat / trace_core_00000000_suite_*.log so the ==="
echo "=== aggregate step below only reflects this 3-seed run, not stale 1-seed leftovers ==="
rm -f "$REPO/cpu"/coverage_suite_*.dat "$REPO/cpu"/trace_core_00000000_suite_*.log

echo ""
echo "=== Running suite against sim_build_opentitan_upstream/Vtop ==="
bash "$REPO/cpu/run_testlist_suite.sh"

echo ""
echo "=== Aggregating coverage ==="
cd "$REPO/cpu"
python3 ../rl-coverage/level11_maxconfig/aggregate_suite_coverage.py
