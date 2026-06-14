"""test_programs.py — testează programe specifice și raportează ce bins noi acoperă."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "level9_ops"))

import starter

baseline_ids, covdat_map, target_bins = starter.setup()

PROGRAMS = {
    "load_use_hazard": [
        (21, 5, 20, 0, 2),   # LW x5, 0(x20)
        (0,  6,  5,  7, 0),  # ADD x6, x5, x7  ← load-use hazard
        (21, 7, 20, 0, 3),   # LW x7, 100(x20)
        (0,  8,  7,  6, 0),  # ADD x8, x7, x6  ← load-use hazard
        (20, 9, 20, 0, 3),   # LH x9, 100(x20)
        (0, 10,  9,  5, 0),  # ADD x10, x9, x5 ← load-use hazard
    ],
    "misaligned_load_x1": [
        # x1=0x00001001 (preloaded, misaligned)
        (21, 5,  1, 0, 2),   # LW x5, 0(x1)   → misaligned → mcause=4
        (21, 6,  1, 0, 3),   # LW x6, 100(x1) → misaligned
        (19, 7,  1, 0, 0),   # LH x7, -2048(x1) → misaligned
    ],
    "exception_diversity": [
        (62, 0, 0, 0, 0),    # ECALL  → mcause=11
        (63, 0, 0, 0, 0),    # EBREAK → mcause=3
        (21, 5,  1, 0, 2),   # LW misaligned → mcause=4
        (62, 0, 0, 0, 0),    # ECALL again
        (10, 5, 16, 0, 0),   # ADDI x5, x16, -2048 (x16=0xFFFFFFFF)
        (62, 0, 0, 0, 0),    # ECALL → mcause=11, mepc=diferit
    ],
    "csrrw_mtvec_diverse": [
        (64, 5,  0, 0, 2),   # LUI x5, 0x12345 → x5=0x12345000
        (27, 6,  5, 0, 1),   # CSRRW x6, mtvec, x5 → mtvec=0x12345000
        (62, 0,  0, 0, 0),   # ECALL → salvează mepc, sare la noul mtvec
        (64, 7,  0, 0, 3),   # LUI x7, 0xFFFFF → x7=0xFFFFF000
        (27, 8,  7, 0, 1),   # CSRRW x8, mtvec, x7
        (62, 0,  0, 0, 0),   # ECALL
        (27, 9,  0, 0, 0),   # CSRRW x9, mtvec, x0 → mtvec=0 (MODE=direct)
    ],
    "counter_inhibit": [
        (27, 5, 16, 0, 0),   # CSRRW x5, mcountinhibit, x16 (0xFFFFFFFF)
        (28, 6,  0, 0, 0),   # CSRRS x6, mcountinhibit, x0  (read back)
        (27, 7,  0, 0, 0),   # CSRRW x7, mcountinhibit, x0  (clear)
        (27, 8, 19, 0, 0),   # CSRRW x8, mcountinhibit, x19 (0x55555555)
    ],
    "many_nops_counter": [
        (10, 0, 0, 0, 2),    # ADDI x0,x0,0 (NOP) × 50
    ] * 50 + [
        (29, 5, 0, 0, 0),    # CSRRC x5, mcycle, x0  (read mcycle)
        (29, 6, 0, 0, 0),    # CSRRC x6, minstret, x0
    ],
}

print(f"\n{'='*60}")
print(f"{'Program':<30} {'new_total':>10} {'new_target':>10}")
print(f"{'='*60}")

for name, actions in PROGRAMS.items():
    r = starter.run_and_check(actions, baseline_ids, covdat_map, target_bins)
    flag = " *** NEW ***" if r['new_total'] > 0 else ""
    print(f"{name:<30} {r['new_total']:>10} {r['new_target']:>10}{flag}")
    if r['new_total'] > 0:
        print(f"  Signals: {r['signals'][:5]}")

print(f"{'='*60}")
