"""codec_l11.py — L10 (87 ops, RV32IMC) + RV32B (Zba/Zbb/Zbs) = 116 ops.

Codec-ul L10 e RV32IMC-only, deci pe configul max (RV32B=RV32BFull) un bloc
mare de bins din ibex_alu / ibex_decoder rămâne neacoperit: tot ce decode-ul
marchează drept legal doar dacă RV32B != RV32BNone (vezi ibex_decoder.sv,
OPCODE_OP cu {instr[31:25],instr[14:12]} pe Zba/Zbb/Zbs, și OPCODE_OP_IMM
funct3 in {001,101} pe variantele *I).

Encodinguri verificate direct din ibex_decoder.sv (RV32BFull):

  Zba (R-type, opcode=0110011):
    SH1ADD  f7=0010000 f3=010
    SH2ADD  f7=0010000 f3=100
    SH3ADD  f7=0010000 f3=110

  Zbb logic + rotate (R-type):
    ANDN    f7=0100000 f3=111
    ORN     f7=0100000 f3=110
    XNOR    f7=0100000 f3=100
    ROL     f7=0110000 f3=001
    ROR     f7=0110000 f3=101

  Zbb min/max (R-type):
    MIN     f7=0000101 f3=100
    MAX     f7=0000101 f3=110
    MINU    f7=0000101 f3=101
    MAXU    f7=0000101 f3=111

  Zbb pack (R-type):
    PACK    f7=0000100 f3=100
    PACKH   f7=0000100 f3=111
    PACKU   f7=0100100 f3=100

  Zbb single-operand (I-type OP-IMM, f3=001, imm12 fix — selectorul e in
  imm[6:0], imm[11:7]=01100 pentru toate):
    CLZ     imm12=0x600
    CTZ     imm12=0x601
    CPOP    imm12=0x602
    SEXT_B  imm12=0x604
    SEXT_H  imm12=0x605

  Zbb rotate immediate (I-type OP-IMM, f3=101, f7=0110000, shamt=imm[4:0]):
    RORI

  Zbs (R-type):
    BCLR    f7=0100100 f3=001
    BSET    f7=0010100 f3=001
    BINV    f7=0110100 f3=001
    BEXT    f7=0100100 f3=101

  Zbs immediate (I-type OP-IMM, shamt=imm[4:0]):
    BCLRI   f3=001 f7=0100100
    BSETI   f3=001 f7=0010100
    BINVI   f3=001 f7=0110100
    BEXTI   f3=101 f7=0100100

Total ops noi: 3 + 5 + 4 + 3 + 5 + 1 + 4 + 4 = 29  →  N_OPS = 87 + 29 = 116.
"""

import sys
from pathlib import Path

_L10 = Path(__file__).resolve().parent.parent / "level10_ops"
sys.path.insert(0, str(_L10))

from codec_l10 import (  # noqa: E402
    N_OPS as L10_N_OPS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    N_CSR_BUCKETS,
    encode as l10_encode,
)

SHAMT_BUCKET_VALUES = [0, 8, 16, 24, 31]
assert len(SHAMT_BUCKET_VALUES) == IMM_BUCKETS

# ── Op indices noi (87..115) ────────────────────────────────────────────────
SH1ADD = 87;  SH2ADD = 88;  SH3ADD = 89
ANDN   = 90;  ORN    = 91;  XNOR   = 92;  ROL = 93;  ROR = 94
MIN    = 95;  MAX    = 96;  MINU   = 97;  MAXU = 98
PACK   = 99;  PACKH  = 100; PACKU  = 101
CLZ    = 102; CTZ    = 103; CPOP   = 104; SEXT_B = 105; SEXT_H = 106
RORI   = 107
BCLR   = 108; BSET   = 109; BINV   = 110; BEXT = 111
BCLRI  = 112; BSETI  = 113; BINVI  = 114; BEXTI = 115

N_OPS = 116

# ── R-type (opcode=0b0110011) ───────────────────────────────────────────────
_R_F3F7 = {
    SH1ADD: (0b010, 0b0010000), SH2ADD: (0b100, 0b0010000), SH3ADD: (0b110, 0b0010000),
    ANDN:   (0b111, 0b0100000), ORN:    (0b110, 0b0100000), XNOR:   (0b100, 0b0100000),
    ROL:    (0b001, 0b0110000), ROR:    (0b101, 0b0110000),
    MIN:    (0b100, 0b0000101), MAX:    (0b110, 0b0000101),
    MINU:   (0b101, 0b0000101), MAXU:   (0b111, 0b0000101),
    PACK:   (0b100, 0b0000100), PACKH:  (0b111, 0b0000100), PACKU: (0b100, 0b0100100),
    BCLR:   (0b001, 0b0100100), BSET:   (0b001, 0b0010100),
    BINV:   (0b001, 0b0110100), BEXT:   (0b101, 0b0100100),
}

# ── I-type shamt (opcode=0b0010011, funct7 fix + imm[4:0]=shamt) ───────────
_I_SHAMT_F3F7 = {
    RORI:  (0b101, 0b0110000),
    BCLRI: (0b001, 0b0100100),
    BSETI: (0b001, 0b0010100),
    BINVI: (0b001, 0b0110100),
    BEXTI: (0b101, 0b0100100),
}

# ── I-type fixed imm12 (opcode=0b0010011, funct3=001, no shamt) ────────────
_I_FIXED12 = {
    CLZ:    0x600,
    CTZ:    0x601,
    CPOP:   0x602,
    SEXT_B: 0x604,
    SEXT_H: 0x605,
}


def _r_type(f7: int, f3: int, rd: int, rs1: int, rs2: int) -> int:
    return (f7 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) \
         | (f3 << 12) | ((rd & 0x1F) << 7) | 0b0110011


def _i_shamt(f7: int, f3: int, rd: int, rs1: int, shamt: int) -> int:
    return (f7 << 25) | ((shamt & 0x1F) << 20) | ((rs1 & 0x1F) << 15) \
         | (f3 << 12) | ((rd & 0x1F) << 7) | 0b0010011


def _i_fixed(imm12: int, rd: int, rs1: int) -> int:
    return ((imm12 & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) \
         | (0b001 << 12) | ((rd & 0x1F) << 7) | 0b0010011


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int, csr_bucket: int = 0) -> int:
    if op_i in _R_F3F7:
        f3, f7 = _R_F3F7[op_i]
        return _r_type(f7, f3, rd, rs1, rs2)
    if op_i in _I_SHAMT_F3F7:
        f3, f7 = _I_SHAMT_F3F7[op_i]
        shamt = SHAMT_BUCKET_VALUES[imm_bucket % len(SHAMT_BUCKET_VALUES)]
        return _i_shamt(f7, f3, rd, rs1, shamt)
    if op_i in _I_FIXED12:
        return _i_fixed(_I_FIXED12[op_i], rd, rs1)
    return l10_encode(op_i, rd, rs1, rs2, imm_bucket, csr_bucket)


def emit_program(actions):
    nop = encode(10, 0, 0, 0, 2)  # ADDI x0, x0, 0
    return [encode(*a) for a in actions] + [nop] * 16


def _self_test():
    for op_i in range(N_OPS):
        for ib in range(IMM_BUCKETS):
            w = encode(op_i, 5, 6, 7, ib)
            assert 0 <= w <= 0xFFFFFFFF, f"op={op_i} ib={ib} -> 0x{w:08x}"

    # R-type: opcode=0110011, verifică (f7,f3) pentru câteva ops
    for op_i, (f3, f7) in _R_F3F7.items():
        w = encode(op_i, 5, 6, 7, 0)
        assert (w & 0x7F) == 0b0110011, f"op={op_i}: opcode != OP, 0x{w:08x}"
        assert ((w >> 12) & 0x7) == f3,  f"op={op_i}: funct3 mismatch, 0x{w:08x}"
        assert ((w >> 25) & 0x7F) == f7, f"op={op_i}: funct7 mismatch, 0x{w:08x}"

    # I-type shamt: opcode=0010011, verifică (f3,f7) + shamt în imm[24:20]
    for op_i, (f3, f7) in _I_SHAMT_F3F7.items():
        w = encode(op_i, 5, 6, 0, 1)  # imm_bucket=1 -> shamt=8
        assert (w & 0x7F) == 0b0010011, f"op={op_i}: opcode != OP-IMM, 0x{w:08x}"
        assert ((w >> 12) & 0x7) == f3,  f"op={op_i}: funct3 mismatch, 0x{w:08x}"
        assert ((w >> 25) & 0x7F) == f7, f"op={op_i}: funct7 mismatch, 0x{w:08x}"
        assert ((w >> 20) & 0x1F) == 8,  f"op={op_i}: shamt mismatch, 0x{w:08x}"

    # I-type fixed imm12 (CLZ/CTZ/CPOP/SEXT.B/SEXT.H)
    for op_i, imm12 in _I_FIXED12.items():
        w = encode(op_i, 5, 6, 0, 0)
        assert (w & 0x7F) == 0b0010011, f"op={op_i}: opcode != OP-IMM, 0x{w:08x}"
        assert ((w >> 12) & 0x7) == 0b001, f"op={op_i}: funct3 != 001, 0x{w:08x}"
        assert ((w >> 20) & 0xFFF) == imm12, f"op={op_i}: imm12 mismatch, 0x{w:08x}"

    # Ops L10 (0..86) trebuie să rămână identice
    for op_i in range(L10_N_OPS):
        for ib in range(IMM_BUCKETS):
            assert encode(op_i, 5, 6, 7, ib) == l10_encode(op_i, 5, 6, 7, ib), \
                f"L10 op={op_i} ib={ib} a fost modificat!"

    print(f"[OK] L11 codec self-test: {N_OPS} ops x {IMM_BUCKETS} buckets "
          f"(L10={L10_N_OPS} + {N_OPS - L10_N_OPS} RV32B)")


if __name__ == "__main__":
    _self_test()
