"""test_rv32b.py — sanity check: rulează un program cu toate cele 29 de
operatii RV32B noi din codec_l11 (Zba/Zbb/Zbs), o singura data, si compara
hits-urile rezultate cu l11_baseline_hits.pkl.

Scop: confirma rapid daca cele 29 ops produc bins NOI (decodate legal pe
RV32B=RV32BFull) sau daca cad pe illegal_insn / bins deja acoperite.

Usage:
    python3 test_rv32b.py [--hits l11_baseline_hits.pkl]
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

THIS   = Path(__file__).resolve().parent
ML4DV  = (THIS.parent.parent / "cpu").resolve()
VTOP   = ML4DV / "sim_build_max" / "Vtop"
COVDAT = ML4DV / "coverage_max.dat"
PROGRAM_JSON = "/tmp/rl_l11_rv32b_test.json"

sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
import cov_parser
import codec_l11 as c11

_F_RE     = re.compile(r"\x01f\x02[^\x01]+")
_N_RE     = re.compile(r"\x01n\x02[^\x01]+")
_H_DOT_RE = re.compile(r"(\x01h\x02)\.")

def norm(key):
    k = _F_RE.sub("", key)
    k = _N_RE.sub("", k)
    k = _H_DOT_RE.sub(r"\1", k)
    return k


def run_words(words):
    with open(PROGRAM_JSON, "w") as f:
        json.dump({"n": len(words), "agent": "l11_rv32b_test",
                   "machine_code": [int(w) for w in words]}, f)

    env = os.environ.copy()
    site_packages = sysconfig.get_paths()["purelib"]
    cocotb_libs = os.path.join(site_packages, "cocotb", "libs")
    env["LD_LIBRARY_PATH"] = (
        cocotb_libs + ":/usr/lib/x86_64-linux-gnu"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = (
        str(ML4DV) + ":" + site_packages
        + ":" + env.get("PYTHONPATH", "")
    )
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = PROGRAM_JSON

    proc = subprocess.run(
        [str(VTOP), f"+verilator+coverage+file+{COVDAT}"],
        cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


# Toate cele 29 ops RV32B noi: codec_l11.N_OPS - codec_l10.N_OPS
_NEW_OPS_START = 87
_NEW_OPS = list(range(_NEW_OPS_START, c11.N_OPS))
_NEW_OP_NAMES = {
    87: "SH1ADD", 88: "SH2ADD", 89: "SH3ADD",
    90: "ANDN", 91: "ORN", 92: "XNOR", 93: "ROL", 94: "ROR",
    95: "MIN", 96: "MAX", 97: "MINU", 98: "MAXU",
    99: "PACK", 100: "PACKH", 101: "PACKU",
    102: "CLZ", 103: "CTZ", 104: "CPOP", 105: "SEXT_B", 106: "SEXT_H",
    107: "RORI",
    108: "BCLR", 109: "BSET", 110: "BINV", 111: "BEXT",
    112: "BCLRI", 113: "BSETI", 114: "BINVI", 115: "BEXTI",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default="l11_baseline_hits.pkl")
    args = ap.parse_args()

    if not VTOP.exists():
        sys.exit(f"Max-config Vtop not found at {VTOP}. Run `cd cpu && make CONFIG=max` first.")

    baseline = set()
    if Path(args.hits).exists():
        with open(args.hits, "rb") as f:
            baseline = pickle.load(f)
        print(f"Baseline incarcat: {len(baseline):,} hits din {args.hits}")
    else:
        print(f"[!] {args.hits} nu exista — comparam fata de set vid")

    # Construim un program: fiecare op nou de cate 5 ori, cu (rd,rs1,rs2)
    # variate ca sa exercitam mai multe registre/valori, plus toate cele
    # 5 imm_buckets pentru variantele *I (RORI/BCLRI/BSETI/BINVI/BEXTI).
    actions = []
    regs = [(5, 6, 7), (10, 11, 12), (15, 16, 17), (20, 21, 22), (25, 26, 27)]
    for op in _NEW_OPS:
        for ib, (rd, rs1, rs2) in enumerate(regs):
            actions.append((op, rd, rs1, rs2, ib, 0))

    words = c11.emit_program(actions)
    print(f"Program: {len(actions)} instructiuni RV32B + 16 NOP, total {len(words)} words\n")

    summary = run_words(words)
    if summary is None:
        sys.exit("Vtop FAILED")

    tog_prefix = "\x01page\x02v_toggle/"
    ep_hits = {norm(k) for k, v in summary.points.items()
               if v > 0 and tog_prefix in ("\x01" + k)}

    toggle_covered, toggle_total = summary.by_kind["toggle"]
    print(f"Coverage program RV32B (izolat): {100.0*toggle_covered/toggle_total:.2f}% "
          f"({toggle_covered:,}/{toggle_total:,})")

    new_hits = ep_hits - baseline
    print(f"Bins noi fata de baseline: {len(new_hits):,}\n")

    if new_hits:
        # Grupare pe modul
        from collections import Counter
        mod_re = re.compile(r"page\x02v_toggle/([^\x01]+)\x01")
        mods = Counter()
        for k in new_hits:
            m = mod_re.search(k)
            if m:
                mods[m.group(1).split("__")[0]] += 1
        print("Bins noi pe modul:")
        for mod, cnt in mods.most_common():
            print(f"  {mod:<30} +{cnt}")
    else:
        print("Niciun bin nou — instructiunile RV32B probabil cad pe paths deja acoperite "
              "(verifica daca sunt tratate ca illegal_insn).")


if __name__ == "__main__":
    main()
