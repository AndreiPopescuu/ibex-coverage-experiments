#!/bin/bash
set -e
source /home/andrei/ibex_env/bin/activate
cd /home/andrei/IBEX/ibex-coverage-experiments/cpu
MODULE=test_run_for_l8 \
RL_L8_JSON=/home/andrei/IBEX/ibex-coverage-experiments/rl-coverage/level11_maxconfig/corpus_max_baseline_v1.json \
PLUSARGS="+verilator+coverage+file+coverage_opentitan_upstream.dat" \
  make -f Makefile.upstream_opentitan
cp trace_core_00000000.log trace_core_00000000_opentitan_upstream.log
echo "BUILD_DONE_MARKER_OK"
