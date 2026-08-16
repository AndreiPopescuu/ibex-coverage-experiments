#!/bin/bash
# run_llm_profile.sh — regenerate corpus, run through Vtop, and report coverage
# for the llm_rtl_directed_test profile ONLY (the CRT built from constrained_llm_l11.py's
# 40 RTL-derived stream builders). Leaves every other profile's corpus/.dat untouched.
#
# Usage:
#   ./run_llm_profile.sh
set -e
PROFILE=llm_rtl_directed_test
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "${IBEX_VENV:-$HOME/ibex_env}/bin/activate"
cd "$SCRIPT_DIR"

echo "=== Generating corpus for $PROFILE ==="
python3 gen_one_profile_corpus.py "$PROFILE"

echo ""
echo "=== Running through Vtop ==="
cd "$REPO/cpu"
if command -v cocotb-config >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(cocotb-config --libpython)"):${LD_LIBRARY_PATH:-}"
fi
for corpus in "$SCRIPT_DIR"/corpus_suite_${PROFILE}_seed*.json; do
    name=$(basename "$corpus" .json); name=${name#corpus_suite_}
    dat="coverage_suite_${name}.dat"
    rm -f "$dat"
    echo "  --- $name ---"
    MODULE=test_run_for_l8 RL_L8_JSON="$corpus" \
        ./sim_build_opentitan_upstream/Vtop "+verilator+coverage+file+${dat}"
done

echo ""
echo "=== Coverage report ==="
cd "$SCRIPT_DIR"
python3 report_profile_coverage.py "$PROFILE"
