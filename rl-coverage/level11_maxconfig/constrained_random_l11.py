"""constrained_random_l11.py — constrained-random instruction generator for
codec_l11 (143 ops), ported from lowRISC's actual riscv-dv generation LOGIC
(not just "inspired by" — verified line-by-line against the real vendored
source), without requiring UVM/VCS/Xcelium.

NOTE: riscv-dv/pygen were investigated but deliberately NOT vendored (see
rl-coverage/level11_maxconfig/TESTLIST_SUITE_SUMMARY.txt) — the real SV/UVM
generator needs a licensed simulator even just to generate instructions, and
pygen (Google's pure-Python reimplementation) emits full .S assembly text
that this project has no assembler to consume (every level here encodes raw
machine-code words directly in Python instead). So this file re-implements
their generation LOGIC in Python against codec_l11's existing action space,
not their SystemVerilog code.

An earlier version of this file GUESSED at riscv-dv's mechanisms (a fixed
P_HAZARD=0.40 chance of reusing the previous instruction's rd, P_ZERO=0.08
chance of forcing x0, P_SAME_SRC=0.08 chance of forcing rs2==rs1, and a
50/50 boundary-biased immediate/CSR-bucket picker) without having actually
read their source. That guess was WRONG, confirmed by reading
vendor/google_riscv-dv/src/isa/riscv_instr.sv and riscv_instr_stream.sv from
github.com/lowRISC/ibex directly:

  - rd/rs1/rs2 are picked UNIFORMLY over all 32 GPRs (including x0) with NO
    special-casing for hazards/zero-register/same-src. `riscv_reg_t` is a
    plain unweighted 32-value enum; the only register constraint in the
    base flow excludes a handful of reserved registers (sp/tp/scratch),
    which don't have an equivalent concept in this project's flat
    instruction-stream codec.
  - Hazards (RAW/WAR/WAW) are NOT a per-instruction probability check —
    they emerge naturally, by accident, only inside DEDICATED streams that
    shrink the candidate register pool to ~6 registers for a contiguous
    block of instructions (riscv_hazard_instr_stream in
    riscv_load_store_instr_lib.sv), making register collisions likely via
    plain birthday-paradox math. See build_hazard_stream() below.
  - General immediates have no corner-value/boundary bias at all — `imm` is
    a plain `rand bit [31:0]` with no `dist{}` weighting in the base flow.
  - Per-test differentiation (why riscv_loop_test is loop-heavy,
    riscv_pmp_basic_test is PMP-CSR-heavy, etc.) does NOT come from a
    static category-weight table either — real riscv-dv's
    `category_dist[category]` defaults to a FLAT ratio (10 for every
    category) unless a test overrides it via `+dist_<cat>=` plusargs, and
    grepping the real testlist.yaml (57 tests) turns up ZERO such
    overrides. Instead, each named test inserts ratio-sized batches of
    self-contained "directed instruction stream" objects (a loop skeleton,
    explicit PMP CSR writes, ...) into an otherwise flat-random base
    stream. See testlist_l11.py's per-profile stream lists for the port of
    that mechanism, and build_loop_stream()/build_pmp_write_stream() below
    for the individual directed-stream ports.

CATEGORY_WEIGHTS below is now uniform (matching category_dist's real
default) rather than the invented 0.45/0.15/0.08/... mix from before.

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
from codec_l11 import N_OPS, IMM_BUCKETS, N_CSR_BUCKETS, L11_CSRS, emit_program  # noqa: E402

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

# Uniform across categories — matches riscv-dv's real category_dist default
# (10 for every category, i.e. flat) confirmed in riscv_instr_gen_config.sv;
# no test in the real testlist.yaml overrides it via +dist_<cat>=. Per-test
# flavor comes from directed-stream insertion instead (see testlist_l11.py).
CATEGORY_WEIGHTS = {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}
assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9

# Register pool size real riscv-dv's riscv_hazard_instr_stream uses to make
# RAW/WAR/WAW collisions likely by accident (birthday-paradox math) — see
# build_hazard_stream() below. Not a per-instruction probability knob.
HAZARD_POOL_SIZE = 6


def _pick_reg(rng):
    """Uniform pick over all 32 GPRs (including x0) — matches riscv-dv's
    real randomize_gpr()/riscv_reg_t default: no forced-reuse, no
    forced-zero, no forced-same-src biasing (verified against source, see
    module docstring)."""
    return int(rng.integers(0, 32))


def _pick_bucket(rng, n):
    """Uniform pick — riscv-dv's general imm/csr fields have no corner-value
    bias in the base flow (verified against source, see module docstring)."""
    return int(rng.integers(0, n))


def sample_action(rng, category_weights=None, categories=None):
    """Sample one (op, rd, rs1, rs2, imm_bucket, csr_bucket) action.

    category_weights/categories default to the module-level CATEGORY_WEIGHTS/
    CATEGORIES so existing callers are unaffected; testlist_l11.py passes
    per-test-profile overrides of category_weights to reshape the op mix
    without duplicating this sampling logic.
    """
    category_weights = category_weights if category_weights is not None else CATEGORY_WEIGHTS
    categories = categories if categories is not None else CATEGORIES

    cat = rng.choice(list(category_weights.keys()), p=list(category_weights.values()))
    op = int(rng.choice(categories[cat]))

    rd  = _pick_reg(rng)
    rs1 = _pick_reg(rng)
    rs2 = _pick_reg(rng)
    imm_bucket = _pick_bucket(rng, IMM_BUCKETS)
    csr_bucket = _pick_bucket(rng, N_CSR_BUCKETS)

    return (op, rd, rs1, rs2, imm_bucket, csr_bucket)


def build_actions(n_actions, seed, category_weights=None, categories=None):
    rng = np.random.default_rng(seed)
    return [sample_action(rng, category_weights, categories) for _ in range(n_actions)]


# ── Op indices used by the directed streams below (from codec_l10.py's Op
#    enum, unchanged in codec_l11) ──────────────────────────────────────────
_ADDI  = 10
_ORI   = 14
_BEQ, _BNE, _BLT, _BGE, _BLTU, _BGEU = 38, 39, 40, 41, 42, 43
_MEM_OPS = [19, 20, 21, 22, 23, 24, 25, 26]           # LB..LHU, SB..SW
_CSR_OPS = [27, 28, 29, 66, 67, 68]                   # CSRRW/S/C[+I]
_CSRRW, _CSRRS = 27, 28

# imm_bucket index whose IMM_BUCKET_VALUES entry is -100 (a backward, in-range
# offset for this project's 5-value fixed imm-bucket table [-2048,-100,0,100,
# 2047] — there is no per-branch label/PC resolution in this flat-instruction-
# stream codec, so "backward branch" here means "a fixed backward byte offset
# bucket", not a resolved jump-to-label the way a real assembler would do it).
_BACK_BUCKET = 1   # -100
_FWD_BUCKET  = 3   # +100


def build_loop_stream(rng, categories=None):
    """Port of riscv_loop_instr.sv's per-instance skeleton (real riscv-dv
    inserts ~200 instances of this into riscv_loop_test's base stream via
    +directed_instr_1=riscv_loop_instr,20 — see testlist_l11.py): counter +
    bound register init, 1-25 random filler instructions as the loop body,
    an update, then a backward branch. This project's codec has no
    label/PC-relative jump resolution (every branch offset is one of 5 fixed
    IMM_BUCKET_VALUES, see codec_l5.py) so this reproduces the STRUCTURAL
    shape (init/body/update/backward-branch) real riscv-dv builds, not a
    functionally-verified counted loop -- consistent with the rest of this
    project, which has no Spike co-simulation to check correctness anyway.
    """
    categories = categories if categories is not None else CATEGORIES
    counter_reg = int(rng.integers(1, 32))
    bound_reg = int(rng.integers(1, 32))
    actions = [
        (_ADDI, counter_reg, 0, 0, 2, 0),   # counter = 0 (bucket 2 -> imm 0)
        (_ADDI, bound_reg, 0, 0, _FWD_BUCKET, 0),  # bound = 100 (loop trip count proxy)
    ]
    n_filler = int(rng.integers(1, 26))
    actions += [sample_action(rng, categories=categories) for _ in range(n_filler)]
    actions.append((_ADDI, counter_reg, counter_reg, 0, _FWD_BUCKET, 0))  # counter update
    branch_op = int(rng.choice([_BEQ, _BNE, _BLT, _BGE, _BLTU, _BGEU]))
    actions.append((branch_op, 0, counter_reg, bound_reg, _BACK_BUCKET, 0))  # branch back
    return actions


def build_pmp_write_stream(rng):
    """Port of riscv_pmp_cfg.sv::gen_pmp_write_test — "a directed PMP test to
    test writes to all the pmpcfg and pmpaddr" CSRs (their comment, verbatim
    intent): one CSR-write instruction per PMP CSR, covering all 20
    pmpcfg0-3/pmpaddr0-15 entries (see codec_l11.py's L11_CSRS layout).
    """
    pmp_start = L11_CSRS.index(0x3A0)
    pmp_count = 20
    return [
        (int(rng.choice(_CSR_OPS)), _pick_reg(rng), _pick_reg(rng), 0,
         _pick_bucket(rng, IMM_BUCKETS), pmp_start + i)
        for i in range(pmp_count)
    ]


def build_pmp_region_exec_stream(rng, n_regions=4):
    """Approximates ibex_make_pmp_region_exec_stream (ibex_directed_instr_lib.sv):
    real version does a targeted read-modify-write of a specific pmpcfgN byte
    lane's X bit via CSRRSI/AND/OR/CSRRW/CSRRS. This ports the read-modify-
    write SHAPE (CSRRS read -> ORI to flip bits -> CSRRW write back) without
    computing the exact byte-lane offset for a specific PMP entry (this
    project's codec has no bit-field-aware CSR model), so it exercises the
    same read-modify-write toggle pattern on random pmpcfg CSRs rather than a
    precisely-targeted single region's executable bit.
    """
    pmp_start = L11_CSRS.index(0x3A0)  # pmpcfg0..3 are the first 4 entries from here
    actions = []
    for _ in range(n_regions):
        csr_bucket = pmp_start + int(rng.integers(0, 4))
        rtmp = int(rng.integers(1, 32))
        actions.append((_CSRRS, rtmp, 0, 0, 0, csr_bucket))               # read
        actions.append((_ORI, rtmp, rtmp, 0, _FWD_BUCKET, 0))             # flip bits
        actions.append((_CSRRW, 0, rtmp, 0, 0, csr_bucket))               # write back
    return actions


def build_load_store_stream(rng, n_accesses=6):
    """Generic random-offset load/store batch off one shared base register.
    Approximates TWO different real riscv-dv streams that this project
    can't compute resolved addresses for (no Spike/address-space model):
    riscv_load_store_rand_instr_stream (used by riscv_rand_jump_test,
    riscv_mmu_stress_test, riscv_unaligned_load_store_test) generically, and
    ibex_cross_pmp_region_mem_access_stream (used by the PMP profile, when
    called after PMP CSR writes) specifically when it's meant to land near a
    configured region boundary — approximated here as maximizing offset
    *bucket* diversity (all 5 IMM_BUCKET_VALUES, including the extremes)
    rather than a resolved boundary address.
    """
    base = int(rng.integers(1, 32))
    actions = [(_ADDI, base, 0, 0, _pick_bucket(rng, IMM_BUCKETS), 0)]
    for _ in range(n_accesses):
        op = int(rng.choice(_MEM_OPS))
        actions.append((op, _pick_reg(rng), base, 0, _pick_bucket(rng, IMM_BUCKETS), 0))
    return actions


def build_jal_stream(rng):
    """Port of riscv_jal_instr (riscv_directed_instr_lib.sv): a single JAL to
    a random target. Real riscv-dv resolves an actual jump target in the
    program's address space; this codec has no label/address resolution
    (see build_loop_stream's docstring), so the "target" here is just one of
    the 5 fixed IMM_BUCKET_VALUES offsets, same limitation as branches.
    """
    return [(44, int(rng.integers(1, 32)), 0, 0, _pick_bucket(rng, IMM_BUCKETS), 0)]  # JAL


def build_hazard_stream(rng, n_actions, categories=None, pool_size=HAZARD_POOL_SIZE):
    """Port of riscv_hazard_instr_stream (riscv_load_store_instr_lib.sv):
    shrink the candidate register pool to `pool_size` registers (drawn once,
    from x1-x31 — real riscv-dv excludes x0/sp/tp/scratch from this pool so
    the collisions are meaningful, not just repeated x0 no-ops) and draw
    every rd/rs1/rs2 in this block from that pool only. RAW/WAR/WAW hazards
    then occur naturally from the shrunk pool, not from a per-instruction
    probability check. Meant to be inserted as a ratio-sized batch into a
    base stream (see testlist_l11.py), like the real directed stream is.
    """
    categories = categories if categories is not None else CATEGORIES
    pool = rng.choice(np.arange(1, 32), size=pool_size, replace=False)
    cat_names = list(categories.keys())
    actions = []
    for _ in range(n_actions):
        cat = rng.choice(cat_names)  # uniform, matching the flat category_dist default
        op = int(rng.choice(categories[cat]))
        rd, rs1, rs2 = (int(rng.choice(pool)) for _ in range(3))
        imm_bucket = _pick_bucket(rng, IMM_BUCKETS)
        csr_bucket = _pick_bucket(rng, N_CSR_BUCKETS)
        actions.append((op, rd, rs1, rs2, imm_bucket, csr_bucket))
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
        "machine_code": machine_code,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {out_path}: {len(actions)} actions -> {len(machine_code)} words "
          f"(seed={args.seed}, uniform category dist, ported from real riscv-dv logic)")


if __name__ == "__main__":
    main()
