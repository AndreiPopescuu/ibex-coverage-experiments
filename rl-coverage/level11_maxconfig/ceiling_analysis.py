"""ceiling_analysis.py -- theoretical toggle-coverage ceiling for the "max"
Ibex build (cocotb_ibex_max.sv: PMP, ICache+scramble, RV32B, lockstep, ...
all on).

Two questions this answers:

1) "Pana la ce procent se poate ajunge?" -- some toggle bins are wired to a
   compile-time constant at the cocotb_ibex_max.sv testbench boundary (an
   input port tied to 1'b0/1'b1/'0, or a lockstep compare that can only fire
   under an injected fault). No instruction sequence the RL agent could ever
   emit can move those bits, so they are excluded from the *reachable* total
   -- this gives a hard ceiling below 100%.

2) "Dupa ce procent devine greu si are nevoie de combinatii ciudate?" --
   cross-references the *reachable* (non-dead) bins against every hit-set
   this project already accumulated on the minimal build (L9/L10 checkpoints,
   which reached ~72% there). Bins whose normalized key is NOT in that union
   needed genuinely new stimulus (PMP setup, ICache eviction/scramble, RV32B,
   misaligned/exception corners, etc.) even on the *old*, smaller config --
   i.e. they're the ones that historically required non-obvious instruction
   sequences rather than falling out of random/curriculum exploration.

Usage:
    cd cpu && make CONFIG=max sim_build_max/Vtop   # once
    python ceiling_analysis.py
"""

import json
import os
import pickle
import re
import subprocess
import sys
import sysconfig
from collections import Counter, defaultdict
from pathlib import Path

THIS  = Path(__file__).resolve().parent
L5    = (THIS.parent / "level5_real_rtl").resolve()
ML4DV = (THIS.parent.parent / "cpu").resolve()
VTOP  = ML4DV / "sim_build_max" / "Vtop"
COVDAT = ML4DV / "coverage_max.dat"
PROGRAM_JSON = "/tmp/rl_l11_ceiling.json"

sys.path.insert(0, str(L5))
import cov_parser

_O_RE = re.compile(r"\x01o\x02([^\x01]+)")
_F_RE = re.compile(r"\x01f\x02[^\x01]+")
_N_RE = re.compile(r"\x01n\x02[^\x01]+")
_H_DOT_RE = re.compile(r"(\x01h\x02)\.")
_PAGE_MOD_RE = re.compile(r"page\x02v_toggle/([^\x01]+)\x01")


def norm_key(key: str) -> str:
    k = _F_RE.sub("", key)
    k = _N_RE.sub("", k)
    k = _H_DOT_RE.sub(r"\1", k)
    return k


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


# Signals tied to a compile-time constant in cpu/cocotb_ibex_max.sv (top-level
# testbench ports fed literal 0/1/'0), plus one lockstep internal that can
# only fire under an injected fault. Matched against the base signal name
# (bit-index suffix "[n]" stripped) at ANY hierarchy depth, since the RTL
# threads these straight through without gating.
DEAD_SIGNALS = {
    "test_en_i":            "tb ties test_en_i = 1'b0",
    "hart_id_i":            "tb ties hart_id_i = 32'b0",
    "instr_err_i":          "tb ties instr_err_i = 1'b0",
    "data_err_i":           "tb ties data_err_i = 1'b0",
    "irq_software_i":       "tb ties irq_software_i = 1'b0",
    "irq_timer_i":          "tb ties irq_timer_i = 1'b0",
    "irq_external_i":       "tb ties irq_external_i = 1'b0",
    "irq_fast_i":           "tb ties irq_fast_i = 15'b0",
    "irq_nm_i":             "tb ties irq_nm_i = 1'b0",
    "scramble_key_valid_i": "tb ties scramble_key_valid_i = 1'b0",
    "scramble_key_i":       "tb ties scramble_key_i = '0",
    "scramble_nonce_i":     "tb ties scramble_nonce_i = '0",
    "debug_req_i":          "tb ties debug_req_i = 1'b0 (halt-request debug entry unreachable; "
                             "ebreak/trigger entry still reachable)",
    "scan_rst_ni":          "tb ties scan_rst_ni = 1'b1 (scan path never selected)",
    "ram_cfg_i":            "tb ties ram_cfg_i = '0",
    "outputs_mismatch":     "lockstep golden-vs-shadow compare; identical RTL/inputs so this "
                             "never fires without fault injection (none exists in this tb)",
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


def load_pkl(path):
    if not path.exists():
        return set()
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    if not VTOP.exists():
        sys.exit(f"Max-config Vtop not found at {VTOP}. Run "
                  f"`cd cpu && make CONFIG=max sim_build_max/Vtop` first.")

    print("Running a trivial (all-NOP) program on the max build to enumerate "
          "every toggle bin...")
    summary = run_program([0x00000013] * 8, timeout=300)
    if summary is None:
        sys.exit("Vtop run failed.")

    tog_prefix = "\x01page\x02v_toggle/"
    all_toggle_keys = [k for k in summary.points if tog_prefix in ("\x01" + k)]
    total = len(all_toggle_keys)
    print(f"Total toggle bins (max config): {total:,}\n")

    dead_keys = set()
    dead_by_family = Counter()
    dead_by_module = Counter()
    total_by_module = Counter()

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
            print(f"  {sig:<22} {n:>5} bins  -- {note}")
    print(f"\n  TOTAL dead:      {len(dead_keys):>6,} / {total:,} bins "
          f"({100.0*len(dead_keys)/total:.2f}%)")
    print(f"  REACHABLE:       {reachable:>6,} / {total:,} bins")
    print(f"\n  >>> Theoretical ceiling on config max: {ceiling_pct:.2f}% <<<\n")

    print("Dead bins by module:")
    for mod, n in dead_by_module.most_common():
        tot = total_by_module[mod]
        print(f"  {mod:<26} {n:>4}/{tot:<5} dead ({100.0*n/tot:5.1f}% of module's bins)")

    # ---- Empirical baseline: what old (minimal-build) corpora already reach ----
    print("\n" + "=" * 72)
    print("EMPIRICAL BASELINE: hits from minimal-build training runs, replayed "
          "by key onto config max")
    print("=" * 72)
    l9  = load_pkl(THIS.parent / "level9_ops"  / "l9_v2_checkpoint_hits.pkl")
    l10 = load_pkl(THIS.parent / "level10_ops" / "l10_checkpoint_hits.pkl")
    l10f = load_pkl(THIS.parent / "level10_ops" / "l10_focused_checkpoint_hits.pkl")
    old_hits = {norm_key(k) for k in (l9 | l10 | l10f)}
    print(f"  L9 hits: {len(l9):,}  L10 hits: {len(l10):,}  L10-focused: {len(l10f):,}"
          f"  union(normalized): {len(old_hits):,}")

    max_keys_norm = {norm_key(k) for k in all_toggle_keys}
    carried_over = old_hits & max_keys_norm
    print(f"  Of those, keys that also exist as bins on config max: {len(carried_over):,}")
    print(f"  -> that alone would already be "
          f"{100.0*len(carried_over)/total:.2f}% of config max toggle coverage, "
          f"for free, without touching PMP/ICache/RV32B/lockstep-specific ops.\n")

    new_only_modules = sorted(m for m in total_by_module
                               if total_by_module[m] > 0 and m not in
                               {"ibex_core", "ibex_alu", "ibex_decoder", "ibex_id_stage",
                                "ibex_ex_block", "ibex_if_stage", "ibex_controller",
                                "ibex_cs_registers", "ibex_csr", "ibex_counter",
                                "ibex_load_store_unit", "ibex_multdiv_fast",
                                "ibex_register_file_ff", "ibex_compressed_decoder",
                                "ibex_wb_stage", "ibex_tracer"})
    print("Modules that are new/absent on the minimal build (need dedicated "
          "stimulus, can't be 'carried over'):")
    for mod in new_only_modules:
        print(f"  {mod:<26} {total_by_module[mod]:>5} bins")


if __name__ == "__main__":
    main()
