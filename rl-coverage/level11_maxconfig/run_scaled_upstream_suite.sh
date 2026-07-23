#!/bin/bash
# run_scaled_upstream_suite.sh — generate corpora using testlist_l11.scaled_seed_counts()
# (upstream-proportional iteration counts, ~150-run budget -> 171 total after floor/rounding),
# run the full suite on sim_build_opentitan_upstream/Vtop, and aggregate coverage.
#
# Resumable: run_testlist_suite.sh (invoked below) skips any test whose
# coverage_suite_<name>.dat already exists, so stopping this script (Ctrl+C / kill /
# TaskStop) and re-invoking it with --resume picks up where it left off instead of
# redoing finished tests.
#
# Usage:
#   bash run_scaled_upstream_suite.sh            # fresh run: regenerate corpora, wipe old .dat, truncate log
#   bash run_scaled_upstream_suite.sh --resume    # keep existing corpora/.dat/log, just continue the suite
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG="$REPO/rl-coverage/level11_maxconfig/_suite_run.log"

if [ "$1" = "--resume" ]; then
    exec >> "$LOG" 2>&1
    echo ""
    echo "=== RESUMING at $(date) ==="
else
    exec > "$LOG" 2>&1
fi
source "${IBEX_VENV:-$HOME/ibex_env}/bin/activate"

cd "$REPO/rl-coverage/level11_maxconfig"
if [ "$1" != "--resume" ]; then
    echo "=== Generating corpora: scaled-upstream seed budget (~150 target, floor 3/profile) ==="
    python3 gen_testlist_corpora.py --clean

    echo ""
    echo "=== Clearing old coverage_suite_*.dat / trace_core_00000000_suite_*.log ==="
    rm -f "$REPO/cpu"/coverage_suite_*.dat "$REPO/cpu"/trace_core_00000000_suite_*.log
fi

echo ""
echo "=== Running suite against sim_build_opentitan_upstream/Vtop ==="
bash "$REPO/cpu/run_testlist_suite.sh"

echo ""
echo "=== Aggregating coverage ==="
cd "$REPO/cpu"
python3 ../rl-coverage/level11_maxconfig/aggregate_suite_coverage.py
