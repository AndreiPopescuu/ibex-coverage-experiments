#!/bin/bash
set -e
source /home/andrei/ibex_env/bin/activate
cd /home/andrei/IBEX/ibex-coverage-experiments/cpu
rm -f coverage_max_baseline.dat
MODULE=test_run_for_l8 \
RL_L8_JSON=/home/andrei/IBEX/ibex-coverage-experiments/rl-coverage/level11_maxconfig/corpus_max_baseline_v1.json \
  ./sim_build_max/Vtop +verilator+coverage+file+coverage_max_baseline.dat
ls -la coverage_max_baseline.dat
