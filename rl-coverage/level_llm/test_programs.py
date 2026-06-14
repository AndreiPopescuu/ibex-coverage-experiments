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
    # WFI — controller FSM: DECODE→WAIT_SLEEP state (RL agent rareori încearcă WFI)
    "wfi_sequence": [
        (71, 0, 0, 0, 0),    # WFI → WAIT_SLEEP (se întoarce imediat dacă mie=0)
        (10, 5, 0, 0, 3),    # ADDI x5, x0, 100
        (71, 0, 0, 0, 0),    # WFI din nou
        (62, 0, 0, 0, 0),    # ECALL
        (71, 0, 0, 0, 0),    # WFI după exception
    ],
    # FENCE.I — flush prefetch buffer + pipeline
    "fence_i_sequence": [
        (72, 0, 0, 0, 0),    # FENCE.I
        (10, 5, 17, 0, 0),   # ADDI x5, x17, -2048
        (72, 0, 0, 0, 0),    # FENCE.I din nou
        (26, 0, 20, 22, 3),  # SW x22, 100(x20)
        (72, 0, 0, 0, 0),    # FENCE.I după store
        (21, 6, 20, 0, 3),   # LW x6, 100(x20)
    ],
    # JALR cu valori diverse de offset — bt_a_operand și JALR offset path
    "jalr_diverse": [
        (64, 5,  0, 0, 2),   # LUI x5, 0x12345
        (65, 1,  5, 0, 2),   # JALR x1, x5, 0    → sare la 0x12345000 (probabil crash → exception)
        (10, 6,  0, 0, 3),   # ADDI x6, x0, 100
        (65, 1,  6, 0, 1),   # JALR x1, x6, -100 → sare la 0
        (62, 0,  0, 0, 0),   # ECALL
    ],
    # CSR read diversity — citim CSRuri diferite după diverse excepții
    "csr_read_after_exc": [
        (62,  0,  0, 0, 0),  # ECALL → mcause=11, mepc=PC
        (28,  5,  0, 0, 0),  # CSRRS x5, ?, x0  (read mepc)
        (28,  6,  0, 0, 0),  # CSRRS x6, ?, x0  (read mcause)
        (21,  7,  1, 0, 2),  # LW x7, 0(x1)  → misaligned → mcause=4, mtval=addr
        (28,  8,  0, 0, 0),  # CSRRS x8, ?, x0  (read mcause=4)
        (28,  9,  0, 0, 0),  # CSRRS x9, ?, x0  (read mtval)
        (63,  0,  0, 0, 0),  # EBREAK → mcause=3
        (28, 10,  0, 0, 0),  # CSRRS x10, ?, x0
    ],
    # Branch-heavy cu valori extreme — bt_a/b_operand upper bits
    "branch_extreme_values": [
        (64,  5,  0, 0, 3),  # LUI x5, 0xFFFFF → x5=0xFFFFF000
        (64,  6,  0, 0, 4),  # LUI x6, 0x80000 → x6=0x80000000
        (38,  0,  5, 6, 4),  # BEQ x5, x6, +16 (not taken)
        (39,  0,  5, 6, 4),  # BNE x5, x6, +16 (taken)
        (40,  0,  6, 5, 3),  # BLT x6, x5, +12 (taken - 0x80000000 < 0xFFFFF000 signed)
        (41,  0,  5, 6, 3),  # BGE x5, x6, +12
        (0,   7,  5, 6, 0),  # ADD x7, x5, x6
        (38,  0,  7, 0, 4),  # BEQ x7, x0, +16
    ],
    # Multe instrucțiuni MUL/DIV cu toate combinațiile de semn
    "muldiv_all_signs": [
        (30,  5, 17, 17, 0), # MUL  x5, INT_MAX, INT_MAX
        (31,  6, 17, 18, 0), # MULH x6, INT_MAX, INT_MIN (signed × signed)
        (32,  7, 18, 17, 0), # MULHSU x7, INT_MIN, INT_MAX
        (33,  8, 16, 16, 0), # MULHU x8, -1, -1 (unsigned)
        (34,  9, 17,  1, 0), # DIV  x9, INT_MAX, 1
        (35, 10, 16,  1, 0), # DIVU x10, -1, 1
        (36, 11, 18,  3, 0), # REM  x11, INT_MIN, x3(=0x3003)
        (37, 12, 16, 19, 0), # REMU x12, -1, 0x55555555
        (30, 13, 16, 18, 0), # MUL  x13, -1, INT_MIN (edge: -1 × MIN)
        (31, 14, 18, 18, 0), # MULH x14, INT_MIN, INT_MIN
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
