#!/bin/bash
set -e
source /home/andrei/ibex_env/bin/activate
cd /home/andrei/IBEX/ibex-coverage-experiments/rl-coverage/level11_maxconfig
python3 constrained_random_l11.py --n-actions 4000 --seed 0 \
  --out corpus_constrained_random_v1.json
