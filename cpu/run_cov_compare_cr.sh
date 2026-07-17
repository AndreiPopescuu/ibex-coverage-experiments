#!/bin/bash
set -e
source /home/andrei/ibex_env/bin/activate
cd /home/andrei/IBEX/ibex-coverage-experiments/rl-coverage/level5_real_rtl
echo "=== fixed corpus (915 instr, deterministic) ==="
python3 cov_parser.py ../../cpu/coverage_opentitan_upstream.dat
echo ""
echo "=== constrained-random (4057 instr, weighted+hazard) ==="
python3 cov_parser.py ../../cpu/coverage_opentitan_constrained_random.dat
