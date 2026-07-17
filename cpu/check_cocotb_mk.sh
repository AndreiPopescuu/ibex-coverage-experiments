#!/bin/bash
source /home/andrei/ibex_env/bin/activate
cd /home/andrei/IBEX/ibex-coverage-experiments/cpu
echo "COCOTB_MAKEFILES: $(cocotb-config --makefiles)"
grep -n "^[a-zA-Z_.]*:" "$(cocotb-config --makefiles)/Makefile.sim" | head -30
