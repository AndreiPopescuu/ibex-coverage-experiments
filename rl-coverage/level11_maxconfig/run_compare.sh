#!/bin/bash
set -e
source /home/andrei/ibex_env/bin/activate
cd /home/andrei/IBEX/ibex-coverage-experiments/rl-coverage/level11_maxconfig
python3 compare_upstream.py
