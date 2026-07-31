#!/bin/bash
# watch_suite_progress_extended.sh — like watch_suite_progress.sh, but for
# run_testlist_suite_parallel.sh (checks suite_extended_run.log, its own
# completion marker) and polls more often for a closer-to-real-time feel
# while a run is in progress.
#
# Usage: bash watch_suite_progress_extended.sh [poll_seconds]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${IBEX_VENV:-$HOME/ibex_env}/bin/activate"
LOG="$REPO/cpu/suite_extended_run.log"
POLL="${1:-30}"
TOTAL=$(ls "$REPO/rl-coverage/level11_maxconfig"/corpus_suite_*.json 2>/dev/null | wc -l)

cd "$REPO/cpu"
while true; do
    DONE=$(ls coverage_suite_*.dat 2>/dev/null | wc -l)
    if [ "$DONE" -gt 0 ]; then
        AGG=$(python3 ../rl-coverage/level11_maxconfig/aggregate_suite_coverage.py 2>/dev/null | grep '%')
        echo "[$(date '+%H:%M:%S')] [$DONE/$TOTAL tests done] $AGG"
    else
        echo "[$(date '+%H:%M:%S')] [0/$TOTAL tests done] no coverage_suite_*.dat yet"
    fi
    if grep -q "Parallel suite run complete" "$LOG" 2>/dev/null; then
        echo "SUITE FINISHED"
        break
    fi
    sleep "$POLL"
done
