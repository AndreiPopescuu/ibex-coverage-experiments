#!/bin/bash
# watch_suite_progress.sh — poll coverage_suite_*.dat every 10 minutes and print
# running progress + aggregate union coverage so far (safe to run alongside an
# in-progress run_testlist_suite.sh — .dat files only appear once a test fully
# finishes, never partially-written). Exits once "Suite run complete" shows up
# in _suite_run.log.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${IBEX_VENV:-$HOME/ibex_env}/bin/activate"
LOG="$REPO/rl-coverage/level11_maxconfig/_suite_run.log"
TOTAL=$(ls "$REPO/rl-coverage/level11_maxconfig"/corpus_suite_*.json 2>/dev/null | wc -l)

cd "$REPO/cpu"
while true; do
    DONE=$(ls coverage_suite_*.dat 2>/dev/null | wc -l)
    if [ "$DONE" -gt 0 ]; then
        AGG=$(python3 ../rl-coverage/level11_maxconfig/aggregate_suite_coverage.py 2>/dev/null | grep '%')
        echo "[$DONE/$TOTAL tests done] $AGG"
    else
        echo "[0/$TOTAL tests done] no coverage_suite_*.dat yet"
    fi
    if grep -q "Suite run complete" "$LOG" 2>/dev/null; then
        echo "SUITE FINISHED"
        break
    fi
    sleep 600
done
