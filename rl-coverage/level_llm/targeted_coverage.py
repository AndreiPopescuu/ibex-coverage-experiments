"""targeted_coverage.py — programe tintite pentru bins data-dependent ramase."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "level9_ops"))
import starter

baseline_ids, covdat_map, target_bins = starter.setup()

# x16=0xFFFFFFFF(-1), x17=0x7FFFFFFF(INT_MAX), x18=0x80000000(INT_MIN)
# x19=0x55555555, x3=0x3003, x20=mem_base
# imm_sel: 0=-2048, 1=-100, 2=0, 3=100, 4=+16(branch)
# LUI imm_sel: 2=0x12345, 3=0xFFFFF, 4=0x80000

PROGRAMS = {
    # quotient[0]: rezultat impartire impar → bitul 0 = 1
    "div_odd_q0": [
        (34,  5, 16, 16, 0),  # DIV x5, -1, -1 → quotient=1 (odd, bit0=1)
        (34,  6, 17, 16, 0),  # DIV x6, INT_MAX, -1 → -(2^31-1) (odd, bit0=1)
        (34,  7, 19, 16, 0),  # DIV x7, 0x55555555, -1 → -0x55555555 (odd)
        (35,  8,  3,  3, 0),  # DIVU x8, 0x3003, 0x3003 → 1 (odd)
        (36,  9, 17,  3, 0),  # REM x9, INT_MAX, 0x3003 → INT_MAX % 0x3003
        (37, 10, 16, 19, 0),  # REMU x10, 0xFFFFFFFF, 0x55555555 → 3 (odd)
        (34, 11,  3, 16, 0),  # DIV x11, 0x3003, -1 → -0x3003 (odd)
        (35, 12, 19,  3, 0),  # DIVU x12, 0x55555555, 0x3003 → quotient
    ],
    # bt_a_operand: JALR cu rs1 avand biti diversi setati → exercita toti bitii
    "jalr_all_bits": [
        (64,  5,  0,  0,  2),  # LUI x5, 0x12345 → x5=0x12345000
        (65,  1,  5,  0,  2),  # JALR x1, x5, 0  → bt_a=0x12345000 (exception)
        (64,  6,  0,  0,  4),  # LUI x6, 0x80000 → x6=0x80000000
        (65,  1,  6,  0,  2),  # JALR x1, x6, 0  → bt_a=0x80000000 (exception)
        (64,  7,  0,  0,  3),  # LUI x7, 0xFFFFF → x7=0xFFFFF000
        (65,  1,  7,  0,  2),  # JALR x1, x7, 0  → bt_a=0xFFFFF000
        (0,   8, 17, 19,  0),  # ADD x8, INT_MAX, 0x55555555 → 0xD5554554
        (65,  1,  8,  0,  2),  # JALR x1, x8, 0  → bt_a=0xD5554554
        (0,   9, 16, 19,  0),  # ADD x9, -1, 0x55555555 → 0x55555554
        (65,  1,  9,  0,  2),  # JALR x1, x9, 0  → bt_a=0x55555554
        (0,  10, 18, 19,  0),  # ADD x10, INT_MIN, 0x55555555 → 0xD5555555
        (65,  1, 10,  0,  2),  # JALR x1, x10, 0 → bt_a=0xD5555555
        (0,  11, 17, 17,  0),  # ADD x11, INT_MAX, INT_MAX → 0xFFFFFFFE
        (65,  1, 11,  0,  2),  # JALR x1, x11, 0 → bt_a=0xFFFFFFFE
        (0,  12, 19, 19,  0),  # ADD x12, 0x55555555, 0x55555555 → 0xAAAAAAAA
        (65,  1, 12,  0,  2),  # JALR x1, x12, 0 → bt_a=0xAAAAAAAA
    ],
    # bt_b_operand: imediat din instructiune (fixat de imm_sel dar mai multi operanzi)
    "branch_diverse_operands": [
        (64,  5,  0,  0,  2),  # LUI x5, 0x12345 → 0x12345000
        (64,  6,  0,  0,  3),  # LUI x6, 0xFFFFF → 0xFFFFF000
        (0,   7,  5,  6,  0),  # ADD x7, x5, x6 → 0x12344000... wait
        (38,  0,  5,  6,  3),  # BEQ x5, x6, +100 (not taken)
        (39,  0,  5,  6,  3),  # BNE x5, x6, +100 (taken → jump forward)
        (0,   8, 16, 17,  0),  # ADD x8, -1, INT_MAX → 0x7FFFFFFE
        (38,  0,  8,  7,  3),  # BEQ x8, x7, +100
        (39,  0,  8,  7,  3),  # BNE x8, x7, +100
        (0,   9, 18, 19,  0),  # ADD x9, INT_MIN, 0x55555555 → 0xD5555555
        (40,  0,  9, 16,  3),  # BLT x9, x16(-1), +100 (signed: 0xD5555555 < -1 → yes)
        (41,  0, 16,  9,  3),  # BGE x16, x9, +100
        (42,  0,  9, 16,  3),  # BLTU x9, x16, +100 (unsigned: 0xD... < 0xFF... → yes)
        (43,  0, 16,  9,  3),  # BGEU x16, x9, +100
    ],
    # operator[6]: ALU opcode cu bitul 6 setat (encodings >= 64)
    "high_opcode_ops": [
        (64,  5,  0,  0,  2),  # LUI x5, 0x12345 (op=64, bit6=1)
        (64,  6,  0,  0,  3),  # LUI x6, 0xFFFFF (op=64)
        (64,  7,  0,  0,  4),  # LUI x7, 0x80000 (op=64)
        (65,  1,  5,  0,  2),  # JALR x1, x5, 0 (op=65, bit6=1)
        (65,  1,  6,  0,  2),  # JALR x1, x6, 0 (op=65)
        (65,  1,  7,  0,  2),  # JALR x1, x7, 0 (op=65)
        (66,  0,  5,  0,  3),  # JAL x0, +100 (op=66, bit6=1)
        (71,  0,  0,  0,  0),  # WFI (op=71, bit6=1)
        (72,  0,  0,  0,  0),  # FENCE.I (op=72, bit6=1)
    ],
}

print(f"\n{'='*65}")
print(f"{'Program':<35} {'new_total':>10} {'new_target':>10}")
print(f"{'='*65}")

total_new = 0
for name, actions in PROGRAMS.items():
    r = starter.run_and_check(actions, baseline_ids, covdat_map, target_bins)
    flag = " *** NEW ***" if r['new_total'] > 0 else ""
    print(f"{name:<35} {r['new_total']:>10} {r['new_target']:>10}{flag}")
    if r['new_total'] > 0:
        print(f"  Signals: {r['signals'][:8]}")
        total_new += r['new_total']

print(f"{'='*65}")
print(f"Total new bins found: {total_new}")
