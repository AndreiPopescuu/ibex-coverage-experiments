#!/bin/bash
# Fast pre-check: elaborate-only (no C++ build) to catch port/param
# mismatches quickly before committing to the slow full Verilator build.
set -e
cd /home/andrei/IBEX/ibex-coverage-experiments/cpu
FILES=$(sed -n '/^VERILOG_SOURCES/,/^$/p' Makefile.upstream | grep -oE '\$\(PWD\)/[^ \\]+' | sed 's#\$(PWD)#'"$(pwd)"'#')
verilator --lint-only -Wno-fatal \
  -DSYNTHESIS=1 -DRVFI=1 \
  +incdir+$(pwd)/src_upstream/lowrisc_dv_dv_fcov_macros_0 \
  +incdir+$(pwd)/src_upstream/lowrisc_prim_util_get_scramble_params_0/rtl \
  +incdir+$(pwd)/src_upstream/lowrisc_prim_util_memload_0/rtl \
  +incdir+$(pwd)/src_upstream/lowrisc_prim_assert_0.1/rtl \
  +incdir+$(pwd)/src_upstream/lowrisc_prim_secded_0.1/rtl \
  +incdir+$(pwd)/src_upstream/lowrisc_ibex_ibex_core_0.1/rtl \
  --top-module cocotb_ibex \
  $FILES
