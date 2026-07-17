#!/bin/bash
set -e
source /home/andrei/ibex_env/bin/activate
cd /home/andrei/IBEX/ibex-coverage-experiments/cpu
rm -f coverage_opentitan_constrained_random.dat
MODULE=test_run_for_l8 \
RL_L8_JSON=/home/andrei/IBEX/ibex-coverage-experiments/rl-coverage/level11_maxconfig/corpus_constrained_random_v1.json \
  ./sim_build_opentitan_upstream/Vtop +verilator+coverage+file+coverage_opentitan_constrained_random.dat
cp trace_core_00000000.log trace_core_00000000_constrained_random.log
ls -la coverage_opentitan_constrained_random.dat
echo "RUN_DONE_MARKER_OK"
