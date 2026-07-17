"""constrained_random_l11.py — constrained-random instruction generator for
codec_l11 (143 ops), built to be as close in SPIRIT as practical to
lowRISC's own riscv-dv generator (vendor/google_riscv-dv/ inside the fresh
upstream vendor pull, see cpu/src_upstream/_UPSTREAM_PROVENANCE.txt) without
requiring UVM/VCS/Xcelium:

  - Weighted op-category selection, mirroring the `dist{}` weighted-random
    philosophy in vendor/google_riscv-dv/src/riscv_instr_gen_config.sv
    (they don't emit pure uniform-random instructions either — arithmetic
    dominates, system/CSR/exception ops are deliberately rare).
  - Deliberate RAW-hazard injection: with probability P_HAZARD, a new
    instruction's rs1/rs2 reuses the PREVIOUS instruction's rd — directly
    targeting the same gpr_hazard=RAW_HAZARD coverpoint defined in
    vendor/google_riscv-dv/src/riscv_instr_cover_group.sv, and the
    RAW_HAZARD cross-coverpoint already defined in this project's own
    cpu/instructions.py Cov enum.
  - Deliberate zero-register (x0 as rd/rs1/rs2) and same-src (rs1==rs2)
    biasing, targeting Cov.ZERO_DST / Cov.ZERO_SRC / Cov.SAME_SRC from
    cpu/instructions.py.
  - Boundary-biased immediate/CSR bucket selection (favor bucket 0 and the
    last bucket over mid-range values) instead of uniform.

This is NOT a port of their SystemVerilog/UVM code (that needs a licensed
simulator) — it's the same generation STRATEGY re-implemented in Python
against codec_l11's existing action space.

Usage:
    python3 constrained_random_l11.py --n-actions 4000 --seed 0 \
        --out corpus_constrained_random_v1.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codec_l11 import N_OPS, IMM_BUCKETS, N_CSR_BUCKETS, emit_program  # noqa: E402

assert N_OPS == 143

# ── Op categories (index ranges), derived from codec_l11.py / codec_l10.py's
#    Op enum — see codec_l10.py for the 0-86 base and codec_l11.py's module
#    docstring for 87-142. ──────────────────────────────────────────────────
CATEGORIES = {
    "alu":        list(range(0, 19)) + list(range(87, 143)),  # R/I-type ALU + all bitmanip
    "load_store": list(range(19, 27)),                        # LB..LHU, SB..SW
    "csr":        [27, 28, 29, 66, 67, 68],                   # CSRRW/S/C[+I]
    "muldiv":     list(range(30, 38)),                        # MUL..REMU
    "branch":     list(range(38, 44)),                        # BEQ..BGEU
    "jump":       [44, 65],                                   # JAL, JALR
    "compressed": list(range(45, 61)) + list(range(73, 83)),  # C_* ops
    "upper_imm":  [61, 64],                                   # AUIPC, LUI
    "system":     [62, 63, 69, 70, 71, 72],                   # ECALL/EBREAK/FENCE/MRET/WFI/FENCE.I
    "exception":  [83, 84, 85, 86],                            # illegal / misaligned load-store
}
assert sorted(sum(CATEGORIES.values(), [])) == list(range(N_OPS)), \
    "CATEGORIES must partition all 143 op indices exactly once"

# Weighted like a riscv-dv test profile: bulk arithmetic, meaningful
# load/store & branch/compressed presence, CSR/system/exception rare but
# non-zero (matches the "small deliberate fraction" philosophy of their
# testlist.yaml categories, e.g. riscv_illegal_instr_test being a distinct,
# lower-iteration-count test rather than blended at high frequency).
CATEGORY_WEIGHTS = {
    "alu": 0.45, "load_store": 0.15, "muldiv": 0.08, "branch": 0.10,
    "jump": 0.03, "compressed": 0.10, "upper_imm": 0.03, "csr": 0.04,
    "system": 0.015, "exception": 0.005,
}
assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9

P_HAZARD   = 0.40   # probability rs1/rs2 deliberately reuses previous rd (RAW hazard)
P_ZERO     = 0.08   # probability a register field is forced to x0
P_SAME_SRC = 0.08   # probability rs2 is forced equal to rs1


def _pick_reg(rng, last_rd, prefer_hazard):
    if prefer_hazard and last_rd is not None and rng.random() < P_HAZARD:
        return last_rd
    if rng.random() < P_ZERO:
        return 0
    return int(rng.integers(1, 32))


def _pick_bucket(rng, n):
    # Boundary-biased: favor the two extreme buckets over uniform mid-range,
    # mirroring riscv-dv's preference for 0/-1/MIN/MAX corner values.
    if rng.random() < 0.5:
        return int(rng.choice([0, n - 1]))
    return int(rng.integers(0, n))


def sample_action(rng, last_rd):
    cat = rng.choice(list(CATEGORY_WEIGHTS.keys()), p=list(CATEGORY_WEIGHTS.values()))
    op = int(rng.choice(CATEGORIES[cat]))

    rd = _pick_reg(rng, last_rd, prefer_hazard=False)
    rs1 = _pick_reg(rng, last_rd, prefer_hazard=True)
    if rng.random() < P_SAME_SRC:
        rs2 = rs1
    else:
        rs2 = _pick_reg(rng, last_rd, prefer_hazard=True)

    imm_bucket = _pick_bucket(rng, IMM_BUCKETS)
    csr_bucket = _pick_bucket(rng, N_CSR_BUCKETS)

    return (op, rd, rs1, rs2, imm_bucket, csr_bucket), rd


def build_actions(n_actions, seed):
    rng = np.random.default_rng(seed)
    actions = []
    last_rd = None
    for _ in range(n_actions):
        action, rd = sample_action(rng, last_rd)
        actions.append(action)
        last_rd = rd
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-actions", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"corpus_constrained_random_seed{args.seed}_n{args.n_actions}.json",
    )

    actions = build_actions(args.n_actions, args.seed)
    machine_code = emit_program(actions)
    payload = {
        "n": len(machine_code),
        "n_actions": len(actions),
        "agent": "constrained_random_v1",
        "seed": args.seed,
        "category_weights": CATEGORY_WEIGHTS,
        "p_hazard": P_HAZARD, "p_zero": P_ZERO, "p_same_src": P_SAME_SRC,
        "machine_code": machine_code,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {out_path}: {len(actions)} actions -> {len(machine_code)} words "
          f"(seed={args.seed}, category weights inspired by riscv-dv)")


if __name__ == "__main__":
    main()
