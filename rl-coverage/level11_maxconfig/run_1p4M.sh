#!/bin/bash
set -u
cd /home/andrei/IBEX/ibex-coverage-experiments/rl-coverage/level11_maxconfig
export TARGET_INSTRUCTIONS=1394060
bash ./run_llm_profile.sh 8
