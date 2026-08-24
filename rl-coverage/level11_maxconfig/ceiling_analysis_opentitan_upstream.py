"""ceiling_analysis_opentitan_upstream.py -- theoretical toggle-coverage ceiling
for the ACTUAL build used by the suite/RL/LLM-CRT comparisons on L11:
cpu/cocotb_ibex_opentitan_upstream.sv, the byte-for-byte lowRISC "opentitan"
preset on fresh vendored RTL (sim_build_opentitan_upstream/Vtop, 38,696
toggle bins). This is a fork of ceiling_analysis.py (which targets the older,
different "max" build / sim_build_max, 33,624 bins) -- kept separate rather
than editing that file in place, since it documents a specific historical
build's ceiling.

Same two questions, same method: run an all-NOP program to enumerate every
toggle bin that exists on this build, then flag which ones are wired to a
compile-time constant at cocotb_ibex_opentitan_upstream.sv's testbench
boundary -- no instruction sequence can ever move those bits, so they set a
hard ceiling below 100% regardless of stimulus-generation method (suite, RL,
or LLM CRT).

Usage (Vtop already built for this project's L11 work):
    python3 ceiling_analysis_opentitan_upstream.py
"""

import json
import os
import re
import subprocess
import sys
import sysconfig
from collections import Counter
from pathlib import Path

THIS  = Path(__file__).resolve().parent
L5    = (THIS.parent / "level5_real_rtl").resolve()
ML4DV = (THIS.parent.parent / "cpu").resolve()
VTOP  = ML4DV / "sim_build_opentitan_upstream" / "Vtop"
COVDAT = ML4DV / "coverage_opentitan_upstream_ceiling.dat"
PROGRAM_JSON = "/tmp/rl_l11_ceiling_opentitan_upstream.json"

sys.path.insert(0, str(L5))
import cov_parser  # noqa: E402

_O_RE = re.compile(r"\x01o\x02([^\x01]+)")
_PAGE_MOD_RE = re.compile(r"page\x02v_toggle/([^\x01]+)\x01")


def module_of(key: str) -> str | None:
    m = _PAGE_MOD_RE.search(key)
    if not m:
        return None
    return m.group(1).split("__")[0]


def signal_of(key: str) -> str | None:
    m = _O_RE.search(key)
    if not m:
        return None
    return m.group(1)


# Ports tied to a compile-time constant in cpu/cocotb_ibex_opentitan_upstream.sv
# (verified by reading that file directly -- see the u_top instantiation).
# Matched against the base signal name (bit-index suffix "[n]" stripped) at
# ANY hierarchy depth, since the RTL threads these straight through ungated.
DEAD_SIGNALS = {
    "test_en_i":              "tb ties test_en_i = 'b0",
    "scan_rst_ni":             "tb ties scan_rst_ni = 1'b1 (scan path never selected)",
    "hart_id_i":               "tb ties hart_id_i = 32'b0",
    "instr_err_i":             "tb ties instr_err_i = 1'b0",
    "data_err_i":              "tb ties data_err_i = 1'b0",
    "irq_software_i":         "tb ties irq_software_i = 1'b0",
    "irq_timer_i":             "tb ties irq_timer_i = 1'b0",
    "irq_external_i":         "tb ties irq_external_i = 1'b0",
    "irq_fast_i":              "tb ties irq_fast_i = 15'b0",
    "irq_nm_i":                "tb ties irq_nm_i = 1'b0",
    "scramble_key_valid_i":   "tb ties scramble_key_valid_i = '0",
    "scramble_key_i":         "tb ties scramble_key_i = '0",
    "scramble_nonce_i":       "tb ties scramble_nonce_i = '0",
    "debug_req_i":             "tb ties debug_req_i = 'b0 (halt-request debug entry "
                               "unreachable; ebreak/trigger entry still reachable)",
    "ram_cfg_icache_tag_i":   "tb ties ram_cfg_icache_tag_i = '{default: RamCfgReqZero}",
    "ram_cfg_icache_data_i":  "tb ties ram_cfg_icache_data_i = '{default: RamCfgReqZero}",
    "outputs_mismatch":       "ibex_lockstep.sv:653 -- shadow-vs-main core compare; identical "
                               "RTL/inputs so this never fires without fault injection (none "
                               "exists in this tb)",
}

_BIT_SUFFIX_RE = re.compile(r"\[\d+\]$")


def base_signal(sig: str) -> str:
    return _BIT_SUFFIX_RE.sub("", sig)


def run_program(words, timeout=300):
    with open(PROGRAM_JSON, "w") as f:
        json.dump({"n": len(words), "agent": "l11_ceiling",
                   "machine_code": [int(w) for w in words]}, f)
    env = os.environ.copy()
    site_packages = sysconfig.get_paths()["purelib"]
    cocotb_libs = os.path.join(site_packages, "cocotb", "libs")
    env["LD_LIBRARY_PATH"] = (cocotb_libs + ":/usr/lib/x86_64-linux-gnu"
                               + ":" + env.get("LD_LIBRARY_PATH", ""))
    env["PYTHONPATH"] = (str(ML4DV) + ":" + site_packages
                          + ":" + env.get("PYTHONPATH", ""))
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = PROGRAM_JSON
    proc = subprocess.run(
        [str(VTOP), f"+verilator+coverage+file+{COVDAT}"],
        cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


def main():
    if not VTOP.exists():
        sys.exit(f"Vtop not found at {VTOP}.")

    print("Running a trivial (all-NOP) program on sim_build_opentitan_upstream "
          "to enumerate every toggle bin...")
    summary = run_program([0x00000013] * 8, timeout=300)
    if summary is None:
        sys.exit("Vtop run failed.")

    tog_prefix = "\x01page\x02v_toggle/"
    all_toggle_keys = [k for k in summary.points if tog_prefix in ("\x01" + k)]
    total = len(all_toggle_keys)
    print(f"Total toggle bins (opentitan_upstream config): {total:,}\n")

    dead_keys = set()
    dead_by_family = Counter()
    dead_by_module = Counter()
    total_by_module = Counter()
    unmatched_signal_bases = Counter()

    for k in all_toggle_keys:
        mod = module_of(k) or "?"
        total_by_module[mod] += 1
        sig = signal_of(k)
        if not sig:
            continue
        base = base_signal(sig)
        if base in DEAD_SIGNALS:
            dead_keys.add(k)
            dead_by_family[base] += 1
            dead_by_module[mod] += 1

    reachable = total - len(dead_keys)
    ceiling_pct = 100.0 * reachable / total

    print("=" * 72)
    print("STRUCTURALLY DEAD BINS (tied to a compile-time constant in the tb)")
    print("=" * 72)
    for sig, note in DEAD_SIGNALS.items():
        n = dead_by_family.get(sig, 0)
        if n:
            print(f"  {sig:<26} {n:>5} bins  -- {note}")
    print(f"\n  TOTAL dead:      {len(dead_keys):>6,} / {total:,} bins "
          f"({100.0*len(dead_keys)/total:.2f}%)")
    print(f"  REACHABLE:       {reachable:>6,} / {total:,} bins")
    print(f"\n  >>> Theoretical ceiling on opentitan_upstream config: {ceiling_pct:.2f}% <<<\n")

    print("Dead bins by module:")
    for mod, n in dead_by_module.most_common():
        tot = total_by_module[mod]
        print(f"  {mod:<26} {n:>4}/{tot:<5} dead ({100.0*n/tot:5.1f}% of module's bins)")

    print("\nAll modules present on this build (bin totals):")
    for mod, tot in total_by_module.most_common():
        print(f"  {mod:<26} {tot:>5} bins")


if __name__ == "__main__":
    main()
