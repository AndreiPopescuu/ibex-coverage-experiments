#!/bin/bash
set -e
source /home/andrei/ibex_env/bin/activate
cd /home/andrei/IBEX/ibex-coverage-experiments/rl-coverage/level5_real_rtl
python3 cov_parser.py ../../cpu/coverage_opentitan_upstream.dat
