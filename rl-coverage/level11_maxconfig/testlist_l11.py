"""testlist_l11.py — maps lowRISC's dv/uvm/core_ibex/riscv_dv_extension/testlist.yaml
(57 named UVM+riscv-dv tests, read directly from github.com/lowRISC/ibex master this
session) onto this project's own Python instruction generator, for the subset of tests
expressible as a pure instruction-mix bias on the flat-instruction-stream cocotb
testbench (cpu/test_run_for_l8.py + codec_l11.py).

No riscv-dv/pygen code is vendored here — their real UVM generator needs a licensed
SystemVerilog+UVM simulator even just to generate instructions, and pygen (Google's
pure-Python reimplementation) emits full .S assembly text that this project has no
assembler to consume (every level here encodes raw machine-code words directly in
Python instead — see codec_l10.py/codec_l11.py). So PROFILES below re-implements each
portable lowRISC test's generation STRATEGY, reusing constrained_random_l11.py's
weighted-category/hazard-injection sampler with per-test category weights.

Every other lowRISC test name that needs infrastructure this testbench doesn't have
(a debug module driving debug_req_i, a randomized interrupt agent, Spike co-simulation,
or force/release fault injection into internal RF/RAM/PC signals) is listed in SKIPPED
with the concrete reason, so the suite accounts for all 57 names instead of silently
dropping the ones it can't do. See TESTLIST_SUITE_SUMMARY.txt for the full writeup.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constrained_random_l11 import CATEGORIES, CATEGORY_WEIGHTS, sample_action  # noqa: E402
from codec_l11 import N_OPS, IMM_BUCKETS, L11_CSRS, N_CSR_BUCKETS, emit_program  # noqa: E402

assert N_OPS == 143

_NOP = 0x00000013  # ADDI x0, x0, 0 — standard RV32 NOP encoding, hardcoded so this file
                    # doesn't need to reach into codec_l11's op-index table for it.

_ALL_CATS = list(CATEGORY_WEIGHTS.keys())


def _weights(**kw):
    """Build a full category_weights dict (all _ALL_CATS keys, unlisted = 0.0)."""
    d = {c: 0.0 for c in _ALL_CATS}
    for k, v in kw.items():
        assert k in d, f"unknown category {k!r} (not in {_ALL_CATS})"
        d[k] = v
    total = sum(d.values())
    assert abs(total - 1.0) < 1e-6, f"weights {kw} sum to {total}, not 1.0"
    return d


@dataclass
class Profile:
    name: str
    description: str
    n_actions: int
    builder: object              # callable(seed) -> list[int] machine-code words
    category_weights: dict = None  # for provenance in the corpus JSON, when applicable
    deterministic: bool = False    # builder ignores seed entirely — running >1 seed is a no-op


def _build_weighted(category_weights, n_actions, categories=None,
                     p_hazard=None, p_zero=None, p_same_src=None):
    def _builder(seed):
        rng = np.random.default_rng(seed)
        kwargs = {}
        if p_hazard is not None:
            kwargs["p_hazard"] = p_hazard
        if p_zero is not None:
            kwargs["p_zero"] = p_zero
        if p_same_src is not None:
            kwargs["p_same_src"] = p_same_src
        actions = []
        last_rd = None
        for _ in range(n_actions):
            action, last_rd = sample_action(rng, last_rd, category_weights, categories, **kwargs)
            actions.append(action)
        return emit_program(actions)
    return _builder


# ─────────────────────────────────────────────────────────────────────────────
# Portable profiles — plain weighted-random instruction mix
# ─────────────────────────────────────────────────────────────────────────────

_BASELINE_WEIGHTS = dict(CATEGORY_WEIGHTS)  # today's default mix (0.45/0.15/0.08/...)

_ARITH_WEIGHTS = _weights(alu=0.65, muldiv=0.20, upper_imm=0.10, csr=0.05)
_JUMP_WEIGHTS = _weights(jump=0.35, alu=0.30, branch=0.15, load_store=0.10,
                          compressed=0.05, upper_imm=0.03, csr=0.02)
_JUMP_STRESS_WEIGHTS = _weights(jump=0.55, alu=0.25, branch=0.10, upper_imm=0.05, compressed=0.05)
_LOOP_WEIGHTS = _weights(branch=0.40, alu=0.35, load_store=0.10, jump=0.05,
                          muldiv=0.05, compressed=0.05)
_MMU_STRESS_WEIGHTS = _weights(load_store=0.55, alu=0.25, branch=0.10,
                                compressed=0.05, upper_imm=0.05)
_ILLEGAL_WEIGHTS = _weights(exception=0.40, alu=0.30, load_store=0.10, branch=0.10,
                             csr=0.05, system=0.05)
_HINT_WEIGHTS = _weights(alu=0.70, load_store=0.10, branch=0.10, compressed=0.10)
_UNALIGNED_WEIGHTS = _weights(load_store=0.70, alu=0.20, branch=0.05, compressed=0.05)
_RV32IM_WEIGHTS = _weights(alu=0.55, load_store=0.15, muldiv=0.08, branch=0.10, jump=0.03,
                            upper_imm=0.03, csr=0.04, system=0.015, exception=0.005)

# bitmanip: split the "alu" category into plain-ALU vs bitmanip so the profile can bias
# heavily toward the 56 bitmanip ops (Zba/Zbb/Zbs + legacy zbp/zbc/zbe/zbf) specifically,
# not just "alu" as a whole (which today also contains base R/I-type ALU, ops 0-19).
_BITMANIP_CATEGORIES = dict(CATEGORIES)
del _BITMANIP_CATEGORIES["alu"]
_BITMANIP_CATEGORIES["alu_base"] = list(range(0, 19))
_BITMANIP_CATEGORIES["bitmanip"] = list(range(87, 143))
_BITMANIP_WEIGHTS = {c: 0.0 for c in _BITMANIP_CATEGORIES}
_BITMANIP_WEIGHTS.update(bitmanip=0.55, alu_base=0.15, load_store=0.10,
                          branch=0.10, compressed=0.05, csr=0.05)
assert abs(sum(_BITMANIP_WEIGHTS.values()) - 1.0) < 1e-6


def _build_csr_sweep(seed):
    """riscv_csr_test: deterministic sweep of every CSR op x every L11_CSRS entry,
    same cycling style as gen_baseline_corpus.py rather than weighted-random — the
    lowRISC test's own description is "test all CSR instructions on all implemented
    CSR registers", which is a coverage sweep, not a random-stress test.
    """
    del seed  # deterministic on purpose
    csr_ops = CATEGORIES["csr"]  # [27, 28, 29, 66, 67, 68] = CSRRW/S/C[+I]
    regs = list(range(1, 32))
    actions = []
    for i in range(N_CSR_BUCKETS):
        op = csr_ops[i % len(csr_ops)]
        rd = regs[i % len(regs)]
        rs1 = regs[(i + 5) % len(regs)]
        actions.append((op, rd, rs1, rs1, i % IMM_BUCKETS, i))
    return emit_program(actions)


# PMP CSR index range within L11_CSRS: pmpcfg0-3 (4) + pmpaddr0-15 (16) = 20 contiguous
# entries, used by _build_pmp_focus below to bias CSR writes toward PMP configuration
# regardless of what the base weighted sampler's uniform csr_bucket pick would choose.
_PMP_START = L11_CSRS.index(0x3A0)
_PMP_COUNT = 20
assert L11_CSRS[_PMP_START:_PMP_START + _PMP_COUNT] == \
    [0x3A0, 0x3A1, 0x3A2, 0x3A3] + [0x3B0 + i for i in range(16)], \
    "L11_CSRS layout changed — update _PMP_START/_PMP_COUNT in testlist_l11.py"

_PMP_WEIGHTS = _weights(csr=0.55, alu=0.20, load_store=0.20, branch=0.05)


def _build_pmp_focus(seed, n_actions=900, p_pmp=0.8):
    """Stands in for lowRISC's riscv_pmp_basic/_disable_all_regions/_out_of_bounds/
    _full_random/_region_exec_test + riscv_epmp_*_test (10 tests total): reweight the
    CSR category heavily, then for any sampled CSR op force its csr_bucket into the
    PMP range with probability p_pmp. This only exercises ibex_pmp's comparator
    TOGGLE coverage — there's no Spike co-simulation here to check the resulting
    access-fault/no-fault decision is architecturally correct, so it's a coverage
    proxy, not a functional PMP test.
    """
    rng = np.random.default_rng(seed)
    csr_ops = set(CATEGORIES["csr"])
    actions = []
    last_rd = None
    for _ in range(n_actions):
        action, last_rd = sample_action(rng, last_rd, _PMP_WEIGHTS)
        op, rd, rs1, rs2, imm_bucket, csr_bucket = action
        if op in csr_ops and rng.random() < p_pmp:
            csr_bucket = _PMP_START + int(rng.integers(0, _PMP_COUNT))
            action = (op, rd, rs1, rs2, imm_bucket, csr_bucket)
        actions.append(action)
    return emit_program(actions)


# Best-effort curated list of standard RISC-V CSR addresses that Ibex's "opentitan"
# config does NOT implement (F/D floating-point, user-mode trap CSRs w/o N ext,
# supervisor-mode CSRs w/o S-mode) — NOT cross-checked bit-for-bit against
# ibex_cs_registers.sv, just standard-RISC-V addresses for extensions/modes Ibex lacks.
_INVALID_CSRS = [
    0x001, 0x002, 0x003,                    # fflags, frm, fcsr — no F/D extension
    0x004, 0x005, 0x040, 0x041, 0x042, 0x043, 0x044,  # user-mode trap CSRs — no N ext
    0x100, 0x104, 0x105, 0x140, 0x141, 0x142, 0x143,  # supervisor CSRs — no S-mode
    0x180,                                   # satp — no S-mode
]


def _encode_csr_raw(csr, rd, rs1, funct3=0b001):
    return (csr << 20) | ((rs1 & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _build_invalid_csr(seed, n_actions=200):
    """riscv_invalid_csr_test: CSRRW to CSR addresses Ibex doesn't implement, which
    bypasses codec_l11's L11_CSRS lookup table on purpose (that table only contains
    known-legal CSRs) — hence the local raw R-type/I-type CSR encoder above instead
    of reusing codec_l11.encode().
    """
    rng = np.random.default_rng(seed)
    words = []
    for i in range(n_actions):
        csr = _INVALID_CSRS[i % len(_INVALID_CSRS)]
        rd = int(rng.integers(1, 32))
        rs1 = int(rng.integers(1, 32))
        words.append(_encode_csr_raw(csr, rd, rs1))
    return words + [_NOP] * 16


PROFILES = [
    Profile("riscv_arithmetic_basic_test",
            "Arithmetic instruction test, no load/store/branch instructions",
            600, _build_weighted(_ARITH_WEIGHTS, 600), _ARITH_WEIGHTS),
    Profile("riscv_machine_mode_rand_test",
            "Machine mode random instruction test (default boot privilege — same mix "
            "as riscv_rand_instr_test, no dedicated mode-switch stimulus)",
            600, _build_weighted(_BASELINE_WEIGHTS, 600), _BASELINE_WEIGHTS),
    Profile("riscv_rand_instr_test",
            "Random instruction stress test",
            800, _build_weighted(_BASELINE_WEIGHTS, 800), _BASELINE_WEIGHTS),
    Profile("riscv_rand_jump_test",
            "Jump among large number of sub-programs (jump-heavy mix)",
            500, _build_weighted(_JUMP_WEIGHTS, 500), _JUMP_WEIGHTS),
    Profile("riscv_jump_stress_test",
            "Stress back to back jump instructions",
            400, _build_weighted(_JUMP_STRESS_WEIGHTS, 400), _JUMP_STRESS_WEIGHTS),
    Profile("riscv_loop_test",
            "Loop test (branch-back heavy mix)",
            500, _build_weighted(_LOOP_WEIGHTS, 500), _LOOP_WEIGHTS),
    Profile("riscv_mmu_stress_test",
            "Load/store heavy mix stressing varied memory access patterns "
            "(Ibex has no MMU — this stresses the load/store unit and PMP address "
            "comparators instead, the closest analogous logic)",
            600, _build_weighted(_MMU_STRESS_WEIGHTS, 600), _MMU_STRESS_WEIGHTS),
    Profile("riscv_illegal_instr_test",
            "Illegal instruction test, verify the processor can detect illegal instructions",
            400, _build_weighted(_ILLEGAL_WEIGHTS, 400), _ILLEGAL_WEIGHTS),
    Profile("riscv_hint_instr_test",
            "HINT-style instruction test (alu ops biased toward rd=x0)",
            400, _build_weighted(_HINT_WEIGHTS, 400, p_zero=0.5), _HINT_WEIGHTS),
    Profile("riscv_csr_test",
            "Test all CSR instructions on all implemented CSR registers (deterministic sweep)",
            N_CSR_BUCKETS, _build_csr_sweep, None, deterministic=True),
    Profile("riscv_unaligned_load_store_test",
            "Load/store heavy mix (proxy for unaligned-address stress — this project's "
            "imm-bucket granularity isn't fine enough to target specific byte offsets, "
            "so this maximizes load/store density instead as the closest achievable proxy)",
            500, _build_weighted(_UNALIGNED_WEIGHTS, 500), _UNALIGNED_WEIGHTS),
    Profile("riscv_rv32im_instr_test",
            "Random instruction test without compressed instructions",
            500, _build_weighted(_RV32IM_WEIGHTS, 500), _RV32IM_WEIGHTS),
    Profile("riscv_bitmanip_full_test",
            "Random instruction test with B extension in full configuration (the 3 "
            "lowRISC bitmanip test variants — full/otearlgrey/balanced — collapse to "
            "this 1 profile since this project's RTL build has one fixed RV32B "
            "parameter, not one per test)",
            700, _build_weighted(_BITMANIP_WEIGHTS, 700, categories=_BITMANIP_CATEGORIES),
            _BITMANIP_WEIGHTS),

    # ── Partial / best-effort — explicitly non-architectural, see docstrings above ──
    Profile("riscv_pmp_suite_test",
            "PARTIAL — stands in for riscv_pmp_basic/_disable_all_regions/"
            "_out_of_bounds/_full_random/_region_exec_test (5) + riscv_epmp_mml/"
            "_mml_execute_only/_mml_read_only/_mmwp/_rlb_test (5) — 10 lowRISC tests "
            "total. CSR-heavy mix biased toward PMP CSRs; exercises ibex_pmp toggle "
            "coverage only, NOT fault-correctness (no Spike cosim).",
            900, _build_pmp_focus, _PMP_WEIGHTS),
    Profile("riscv_invalid_csr_test",
            "PARTIAL — CSR accesses to a curated list of CSR addresses Ibex's "
            "opentitan config doesn't implement (F/D, user-trap, supervisor-mode). "
            "Curated by extension/mode reasoning, not cross-checked against "
            "ibex_cs_registers.sv line-by-line.",
            200, _build_invalid_csr, None),
    Profile("riscv_user_mode_rand_test",
            "EXPERIMENTAL/lower-priority — no directed mstatus.MPP+MRET prologue is "
            "implemented in this pass (same shape as the PMP prologue added in "
            "commit a2bc7d3 and reverted in c430fa4 for env_l11.py's RL-training use "
            "case; that revert doesn't block a directed prologue HERE, in a separate "
            "test-suite tool with a different goal, but it's flagged lower priority "
            "given the closest precedent was walked back once already). Currently "
            "just the baseline mix with MRET/system ops present incidentally — no "
            "privilege level is actually verified to change.",
            500, _build_weighted(_BASELINE_WEIGHTS, 500), _BASELINE_WEIGHTS),
]

# ─────────────────────────────────────────────────────────────────────────────
# Skipped — every other lowRISC testlist.yaml name, with the concrete reason this
# testbench can't produce equivalent stimulus.
# ─────────────────────────────────────────────────────────────────────────────

_NEEDS_DEBUG = "needs a debug module driving debug_req_i; not present in this testbench"
_NEEDS_IRQ = "needs a randomized interrupt-driving agent; not present in this testbench"
_NEEDS_FAULT_INJECT = ("needs force/release fault injection into internal RF/RAM/PC/icache "
                        "signals via cocotb; not implemented")

SKIPPED = [
    ("riscv_debug_basic_test", _NEEDS_DEBUG),
    ("riscv_debug_triggers_test", _NEEDS_DEBUG),
    ("riscv_debug_stress_test", _NEEDS_DEBUG),
    ("riscv_debug_branch_jump_test", _NEEDS_DEBUG),
    ("riscv_debug_instr_test", _NEEDS_DEBUG),
    ("riscv_debug_wfi_test", _NEEDS_DEBUG),
    ("riscv_dret_test", _NEEDS_DEBUG),
    ("riscv_debug_ebreak_test", _NEEDS_DEBUG),
    ("riscv_debug_ebreakmu_test", _NEEDS_DEBUG),
    ("riscv_debug_csr_entry_test", _NEEDS_DEBUG),
    ("riscv_debug_single_step_test", _NEEDS_DEBUG),
    ("riscv_irq_in_debug_mode_test", _NEEDS_DEBUG + " + " + _NEEDS_IRQ),
    ("riscv_debug_in_irq_test", _NEEDS_DEBUG + " + " + _NEEDS_IRQ),
    ("riscv_assorted_traps_interrupts_debug_test", _NEEDS_DEBUG + " + " + _NEEDS_IRQ),
    ("riscv_single_interrupt_test", _NEEDS_IRQ),
    ("riscv_multiple_interrupt_test", _NEEDS_IRQ),
    ("riscv_nested_interrupt_test", _NEEDS_IRQ),
    ("riscv_interrupt_instr_test", _NEEDS_IRQ),
    ("riscv_interrupt_wfi_test", _NEEDS_IRQ),
    ("riscv_interrupt_csr_test", _NEEDS_IRQ),
    ("riscv_mem_error_test", "needs randomized memory-interface error injection; not implemented"),
    ("riscv_mem_intg_error_test", "needs randomized memory-integrity error injection; not implemented"),
    ("riscv_reset_test", "needs mid-program reset assertion; not supported by the current cocotb flow"),
    ("riscv_pc_intg_test", _NEEDS_FAULT_INJECT),
    ("riscv_rf_intg_test", _NEEDS_FAULT_INJECT),
    ("riscv_rf_addr_intg_test", _NEEDS_FAULT_INJECT),
    ("riscv_ram_intg_test", _NEEDS_FAULT_INJECT),
    ("riscv_icache_intg_test", _NEEDS_FAULT_INJECT),
    ("riscv_ebreak_test", "EBREAK already fires incidentally via the 'system' category in every "
                           "profile; a dedicated profile adds nothing since there's no Spike "
                           "cosim to check the resulting exception is architecturally correct"),
    ("riscv_umode_tw_test", "needs real U-mode entry + mstatus.tw + WFI-traps-to-M-mode "
                             "verification; beyond what the best-effort riscv_user_mode_rand_test "
                             "profile attempts (that one doesn't verify architectural behavior "
                             "either — see its description)"),
]

assert len({p.name for p in PROFILES} | {n for n, _ in SKIPPED}) == len(PROFILES) + len(SKIPPED), \
    "duplicate test name across PROFILES/SKIPPED"


# ─────────────────────────────────────────────────────────────────────────────
# Upstream `iterations:` counts (dv/uvm/core_ibex/riscv_dv_extension/testlist.yaml,
# fetched live from github.com/lowRISC/ibex at commit 8ed87e07 on 2026-07-23 — same
# commit already vendored into cpu/src_upstream/, confirmed byte-identical to master
# at fetch time). Grouped/summed for the 3 profiles here that stand in for more than
# one upstream test name (see PROFILES descriptions above): bitmanip_full_test =
# bitmanip_full(10) + bitmanip_otearlgrey(10) + bitmanip_balanced(10); pmp_suite_test =
# pmp_basic(50) + pmp_disable_all_regions(50) + pmp_out_of_bounds(50) +
# pmp_full_random(600) + pmp_region_exec(20) + epmp_mml(20) + epmp_mml_execute_only(20)
# + epmp_mml_read_only(20) + epmp_mmwp(20) + epmp_rlb(20) = 870. Running these raw
# counts as n_seeds/profile would mean ~1026 Verilator runs total (~870 of them just
# for PMP alone) — 25-43h wall clock, all to fuzz one already-partial/non-architectural
# proxy. UPSTREAM_ITERATIONS is kept as the honest source number for provenance; use
# scaled_seed_counts() below to get a run budget that fits in a few hours while still
# preserving upstream's *relative* emphasis across profiles.
UPSTREAM_ITERATIONS = {
    "riscv_arithmetic_basic_test": 10,
    "riscv_machine_mode_rand_test": 10,
    "riscv_rand_instr_test": 10,
    "riscv_rand_jump_test": 10,
    "riscv_jump_stress_test": 10,
    "riscv_loop_test": 10,
    "riscv_mmu_stress_test": 10,
    "riscv_illegal_instr_test": 15,
    "riscv_hint_instr_test": 10,
    "riscv_csr_test": 5,  # deterministic here regardless (fixed sweep, seed is a no-op)
    "riscv_unaligned_load_store_test": 5,
    "riscv_rv32im_instr_test": 5,
    "riscv_bitmanip_full_test": 30,
    "riscv_pmp_suite_test": 870,
    "riscv_invalid_csr_test": 10,
    "riscv_user_mode_rand_test": 10,
}
assert set(UPSTREAM_ITERATIONS) == {p.name for p in PROFILES}, \
    "UPSTREAM_ITERATIONS must have one entry per PROFILES entry"


def scaled_seed_counts(target_total=150, floor=3):
    """Scale UPSTREAM_ITERATIONS down to ~target_total total non-deterministic runs,
    preserving each profile's iteration count *relative to the others* (so PMP still
    gets noticeably more seeds than e.g. riscv_loop_test, same as upstream intends —
    it's just bounded to something runnable). `floor` guarantees every non-deterministic
    profile gets at least a few seeds so none collapse to 1 sample. Deterministic
    profiles (riscv_csr_test) always map to 1 regardless of target_total/floor, since
    re-running an identical seed-independent sweep is a no-op.

    Returns {profile_name: n_seeds}.
    """
    det_names = {p.name for p in PROFILES if p.deterministic}
    nondet_raw = {n: it for n, it in UPSTREAM_ITERATIONS.items() if n not in det_names}
    raw_total = sum(nondet_raw.values())
    scale = target_total / raw_total
    out = {n: max(floor, round(it * scale)) for n, it in nondet_raw.items()}
    out.update({n: 1 for n in det_names})
    return out


if __name__ == "__main__":
    print(f"{len(PROFILES)} portable/partial profiles (3 of which each stand in for "
          f"multiple lowRISC test names — see descriptions), {len(SKIPPED)} skipped "
          f"lowRISC test names with reasons.")
    for p in PROFILES:
        words = p.builder(0)
        print(f"  [{len(words):5d} words] {p.name:32s} {p.description[:70]}")
    print()
    for name, reason in SKIPPED:
        print(f"  [SKIP] {name:40s} {reason}")
