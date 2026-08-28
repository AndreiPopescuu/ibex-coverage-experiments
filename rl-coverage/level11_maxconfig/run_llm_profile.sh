#!/bin/bash
# run_llm_profile.sh — regenerate corpus, run through Vtop (in parallel), and report
# coverage for the llm_rtl_directed_test profile ONLY (the CRT built from
# constrained_llm_l11.py's 40 RTL-derived stream builders). Leaves every other
# profile's corpus/.dat untouched.
#
# Seed count is picked so total instructions match the full 171-test suite's total
# (139406, summed from this project's existing corpus_suite_*.json — see
# gen_one_profile_corpus.py's --target-instructions), for a fair apples-to-apples
# coverage comparison instead of comparing a tiny corpus against the full suite's
# union. At ~1050 words/seed that's ~133 seeds — WAY more than the 3 seeds the
# default scaled-upstream budget would give this profile, so this runs in parallel
# (like cpu/run_testlist_suite_parallel.sh) instead of one Vtop at a time, or this
# would take hours.
#
# Usage:
#   ./run_llm_profile.sh [N_PARALLEL]
#   ./run_llm_profile.sh 24
set -u
PROFILE=llm_rtl_directed_test
TARGET_INSTRUCTIONS="${TARGET_INSTRUCTIONS:-139406}"
N_PARALLEL="${1:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "${IBEX_VENV:-$HOME/ibex_env}/bin/activate"
cd "$SCRIPT_DIR"

echo "=== Cleaning up stale seed files from any earlier run of this profile ==="
# Without this, leftover corpus_suite_${PROFILE}_seed*.json / matching .dat
# files from a PRIOR run at a different budget/stream-set (e.g. run_1p4M.sh's
# 10x-budget run, which leaves seeds up into the hundreds) silently get
# swept up by this script's own globs below, mixing old and new stream
# content into one contaminated "union". Delete both before regenerating.
rm -f "$SCRIPT_DIR"/corpus_suite_${PROFILE}_seed*.json
rm -f "$REPO/cpu"/coverage_suite_${PROFILE}_seed*.dat

echo "=== Generating corpus for $PROFILE (target ~$TARGET_INSTRUCTIONS instructions) ==="
python3 gen_one_profile_corpus.py "$PROFILE" --target-instructions "$TARGET_INSTRUCTIONS"

echo ""
echo "=== Running through Vtop, $N_PARALLEL in parallel ==="
cd "$REPO/cpu"
if command -v cocotb-config >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(cocotb-config --libpython)"):${LD_LIBRARY_PATH:-}"
fi
export PYGPI_PYTHON_BIN="$(python3 -c 'import sys; print(sys.executable)')"
SIM_BUILD=./sim_build_opentitan_upstream
COV_PARSER="$REPO/rl-coverage/level5_real_rtl/cov_parser.py"
export SIM_BUILD COV_PARSER

run_one() {
    corpus="$1"
    name=$(basename "$corpus" .json); name=${name#corpus_suite_}
    dat="coverage_suite_${name}.dat"
    rm -f "$dat"
    echo "=== $name === starting"
    if COCOTB_TEST_MODULES=test_run_for_l8 RL_L8_JSON="$corpus" \
        "$SIM_BUILD/Vtop" "+verilator+coverage+file+${dat}" \
        >/dev/null 2>&1; then
        if [ -f "$dat" ]; then
            pct=$(python3 "$COV_PARSER" "$dat" 2>/dev/null | grep -E '^\s+(toggle|branch|line)\s*:' | tr '\n' ' ')
            echo "=== $name === OK   $pct"
        else
            echo "=== $name === WARNING: reported success but $dat missing"
        fi
    else
        echo "=== $name === FAILED"
    fi
}
export -f run_one

printf '%s\n' "$SCRIPT_DIR"/corpus_suite_${PROFILE}_seed*.json \
    | xargs -P "$N_PARALLEL" -I{} bash -c 'run_one "$@"' _ {}

echo ""
echo "=== Coverage report ==="
cd "$SCRIPT_DIR"
python3 report_profile_coverage.py "$PROFILE"
