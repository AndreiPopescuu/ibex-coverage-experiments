"""examples.py — Example programs for each group in accessible_bins_for_llm.txt.

These are starting points — not optimized. Use them as reference or replace
entirely with LLM-generated programs.

Usage:
    python examples.py              # run all groups
    python examples.py --group mtvec
"""

import sys, argparse, time
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from starter import (
    run_and_check, setup,
    LUI, AUIPC, ADDI, CSRRW, CSRRS,
    ECALL, EBREAK, MRET,
    MUL, DIV, MULH,
    LB, LH, SB, SH,
    NOP,
)

COVDAT = THIS.parent.parent / "cpu" / "coverage.dat"


def prog_mcountinhibit(n=50):
    """Group 1: mcountinhibit CSR bits [1:31]"""
    acts = []
    CSR_MCOUNTINHIBIT = 22
    for _ in range(n // 10):
        for imm_b in range(5):
            acts += [
                (LUI,   1, 0, 0, imm_b),
                (CSRRW, 0, 1, 0, CSR_MCOUNTINHIBIT),
                (ADDI,  1, 0, 0, 0),
                (CSRRW, 0, 1, 0, CSR_MCOUNTINHIBIT),
            ]
    return acts


def prog_mtvec_bits(n=50):
    """Group 2: mtvec bits [7:1]"""
    acts = []
    CSR_MTVEC = 5
    for _ in range(n // 5):
        for imm_b in range(5):
            acts += [
                (LUI,   1, 0, 0, imm_b),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
                (ADDI,  1, 0, 0, 0),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
                (ADDI,  1, 0, 0, 1),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
                (ADDI,  1, 0, 0, 0),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
            ]
    return acts


def prog_exception_paths(n=60):
    """Group 3: exception paths"""
    acts = []
    for _ in range(n // 6):
        acts += [
            (ECALL,  0, 0, 0, 0),
            (EBREAK, 0, 0, 0, 0),
            NOP, NOP,
            (MRET,   0, 0, 0, 0),
            NOP,
        ]
    return acts


def prog_large_immediates(n=60):
    """Group 4: immediate upper bits — LUI/AUIPC"""
    acts = []
    for _ in range(n // 10):
        for ib in range(5):
            acts += [
                (LUI,   1, 0, 0, ib),
                (AUIPC, 2, 0, 0, ib),
            ]
    return acts[:n]


def prog_csr_read_data(n=50):
    """Group 5: CSR read data bits"""
    acts = []
    CSR_MTVEC   = 5
    CSR_MSTATUS = 0
    CSR_MIE     = 4
    for _ in range(n // 8):
        for ib in range(5):
            acts += [
                (LUI,   1, 0, 0, ib),
                (CSRRW, 2, 1, 0, CSR_MTVEC),
                (CSRRS, 3, 0, 0, CSR_MSTATUS),
                (CSRRS, 4, 0, 0, CSR_MIE),
                NOP, NOP,
                (ADDI,  1, 0, 0, 0),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
            ]
    return acts


def prog_alu_decoder(n=60):
    """Group 6: ALU/decoder — MUL/DIV"""
    acts = []
    for _ in range(n // 6):
        for rs1 in range(1, 5):
            for rs2 in range(1, 5):
                acts += [
                    (MUL,  1, rs1, rs2, 0),
                    (DIV,  2, rs1, rs2, 0),
                    (MULH, 3, rs1, rs2, 0),
                ]
                if len(acts) >= n:
                    break
    return acts[:n]


def prog_data_alignment(n=40):
    """Group 7: data address alignment"""
    acts = []
    for _ in range(n // 4):
        acts += [
            (LB, 1, 0, 0, 1),
            (SB, 1, 0, 0, 1),
            (LH, 1, 0, 0, 2),
            (SH, 1, 0, 0, 2),
        ]
    return acts


PROGRAMS = {
    "mcountinhibit": (prog_mcountinhibit,   "mcountinhibit CSR bits"),
    "mtvec":         (prog_mtvec_bits,       "mtvec low bits"),
    "exception":     (prog_exception_paths,  "exception paths"),
    "immediates":    (prog_large_immediates, "large immediate bits"),
    "csr_read":      (prog_csr_read_data,    "CSR read data"),
    "alu":           (prog_alu_decoder,      "ALU/decoder signals"),
    "alignment":     (prog_data_alignment,   "data address alignment"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group",  default=None)
    ap.add_argument("--covdat", default=str(COVDAT))
    args = ap.parse_args()

    baseline_ids, covdat_map, target_bins = setup(args.covdat)

    groups = [args.group] if args.group else list(PROGRAMS.keys())
    total_new = 0
    cumulative = set()

    for name in groups:
        if name not in PROGRAMS:
            print(f"Unknown group: {name}")
            continue
        fn, desc = PROGRAMS[name]
        print(f"▶ {desc}")
        t0 = time.time()
        actions = fn()
        result  = run_and_check(actions, baseline_ids | cumulative, covdat_map, target_bins)
        print(f"  New bins (total):  {result['new_total']}")
        print(f"  New bins (target): {result['new_target']}")
        for sig in result['target_signals']:
            print(f"    ✅ {sig}")
            cumulative.add(sig)
        if not result['target_signals']:
            print(f"    (none from target list)")
        print(f"  Time: {time.time()-t0:.1f}s\n")
        total_new += result['new_target']

    covered = 14404 + total_new
    pct = 100 * covered / 20023
    print("=" * 60)
    print(f"RESULT: {total_new} new bins from target list")
    print(f"Coverage: {covered:,} / 20,023 = {pct:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
