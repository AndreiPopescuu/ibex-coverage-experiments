#!/bin/bash
# run_testlist_suite_parallel.sh — same as run_testlist_suite.sh but runs
# tests concurrently via xargs -P, instead of one Vtop at a time. Safe
# because each test is a fully independent Vtop invocation writing to its
# own coverage_suite_<name>.dat — exactly the isolation env_l11.py's
# SubprocVecEnv already relies on for RL training (n_envs parallel Vtop
# instances from this same cpu/ cwd, for the whole training session).
#
# Skips the trace_core_00000000.log copy step from the sequential script:
# that filename is fixed/shared across every Vtop instance (not
# parameterizable), so copying it under concurrency would race between
# workers. It's a debug-only execution trace, not used for coverage, so
# dropping it here is safe.
#
# Usage:
#   ./run_testlist_suite_parallel.sh [N_PARALLEL] [--force]
#   ./run_testlist_suite_parallel.sh 24
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${IBEX_VENV:-$HOME/ibex_env}/bin/activate"
cd "$SCRIPT_DIR"

if command -v cocotb-config >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="$(dirname "$(cocotb-config --libpython)"):${LD_LIBRARY_PATH:-}"
fi

N_PARALLEL="${1:-8}"
FORCE=0
[ "${2:-}" = "--force" ] && FORCE=1
export FORCE

CORPUS_DIR="$SCRIPT_DIR/../rl-coverage/level11_maxconfig"
SIM_BUILD=./sim_build_opentitan_upstream
export SIM_BUILD

if [ ! -x "$SIM_BUILD/Vtop" ]; then
    echo "ERROR: $SIM_BUILD/Vtop not found — build it first with:"
    echo "  make -f Makefile.upstream_opentitan"
    exit 1
fi

shopt -s nullglob
CORPORA=("$CORPUS_DIR"/corpus_suite_*.json)
if [ ${#CORPORA[@]} -eq 0 ]; then
    echo "ERROR: no corpus_suite_*.json found in $CORPUS_DIR — run gen_testlist_corpora.py first"
    exit 1
fi

echo "=== Running ${#CORPORA[@]} tests, $N_PARALLEL in parallel ==="

run_one() {
    corpus="$1"
    name=$(basename "$corpus" .json)
    name=${name#corpus_suite_}
    dat="coverage_suite_${name}.dat"

    if [ "$FORCE" -eq 0 ] && [ -s "$dat" ]; then
        echo "=== $name === SKIP (already done)"
        return 0
    fi

    echo "=== $name === starting"
    rm -f "$dat"
    if MODULE=test_run_for_l8 RL_L8_JSON="$corpus" \
        "$SIM_BUILD/Vtop" "+verilator+coverage+file+${dat}" \
        >/dev/null 2>&1; then
        if [ -f "$dat" ]; then
            echo "=== $name === OK"
        else
            echo "=== $name === WARNING: reported success but $dat missing"
        fi
    else
        echo "=== $name === FAILED"
    fi
}
export -f run_one

printf '%s\n' "${CORPORA[@]}" | xargs -P "$N_PARALLEL" -I{} bash -c 'run_one "$@"' _ {}

echo ""
echo "=== Parallel suite run complete ==="
echo "Run: python3 ../rl-coverage/level11_maxconfig/aggregate_suite_coverage.py"
