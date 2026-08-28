"""codec_l11.py — L10 (87 ops, RV32IMC) + RV32B (Zba/Zbb/Zbs = 116 ops,
+ zbp/zbc/zbe/zbf legacy draft ops = 143 ops).

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

Total ops noi (Zba/Zbb/Zbs): 3 + 5 + 4 + 3 + 5 + 1 + 4 + 4 = 29  →  116 ops.

Restul RV32BFull (draft legacy bitmanip, RV32BOTEarlGrey ⊇ RV32BFull pt. astea):
zbp/zbc/zbe/zbf, legal doar cand RV32B in {RV32BOTEarlGrey, RV32BFull}, cu
excepția BFP (legal pt. orice RV32B != RV32BNone) și BCOMPRESS/BDECOMPRESS
(legal doar RV32B == RV32BFull). Encodinguri din ibex_decoder.sv, blocurile
illegal_insn pt. OPCODE_OP ({instr[31:25],instr[14:12]}) și OPCODE_OP_IMM
(f3 in {001,101}, unique case(instr[31:27]) [+ instr[26:20] pt. crc32]).

  zbp (R-type):
    SLO      f7=0010000 f3=001      SRO      f7=0010000 f3=101
    GREV     f7=0110100 f3=101      GORC     f7=0010100 f3=101
    SHFL     f7=0000100 f3=001      UNSHFL   f7=0000100 f3=101
    XPERM.N  f7=0010100 f3=010      XPERM.B  f7=0010100 f3=100
    XPERM.H  f7=0010100 f3=110

  zbp immediate (I-type OP-IMM, shamt=imm[4:0]; instr[25] e don't-care
  in decoder, fixat pe 0 aici):
    SLOI     f3=001 f7=0010000      SROI     f3=101 f7=0010000
    GREVI    f3=101 f7=0110100      GORCI    f3=101 f7=0010100
    SHFLI    f3=001 f7=0000100      UNSHFLI  f3=101 f7=0000100

  zbc (R-type, carry-less multiply):
    CLMUL    f7=0000101 f3=001
    CLMULR   f7=0000101 f3=010
    CLMULH   f7=0000101 f3=011

  zbe (R-type):
    BCOMPRESS    f7=0000100 f3=110
    BDECOMPRESS  f7=0100100 f3=110

  zbf (R-type, bit-field place):
    BFP      f7=0100100 f3=111

  crc32/crc32c (I-type OP-IMM, f3=001, imm12 fix — grupul CLZ, imm[11:7]=
  01100, imm[6:5]=00, imm[4:0]=selector):
    CRC32.B   imm12=0x610   CRC32C.B  imm12=0x618
    CRC32.H   imm12=0x611   CRC32C.H  imm12=0x619
    CRC32.W   imm12=0x612   CRC32C.W  imm12=0x61A

Total ops noi (zbp/zbc/zbe/zbf): 9 + 6 + 3 + 2 + 1 + 6 = 27  →  143 ops.

Rămân neadăugate (necesită rs3, deci extindere de action-space, nu doar
codec): CMIX, CMOV, FSL, FSR (R-type, {instr[26],instr[13:12]}=={1,2'b01})
și FSRI (OP-IMM, f3=101, instr[26]=1).

Zcmp (RV32ZcaZcmp, ibex_compressed_decoder.sv funct3=101 C2 quadrant):
  cm.push   — 5'b11000 in casez(instr[12:8]): exercitează CmPushStoreReg + CmPushDecrSp
  cm.pop    — 5'b11010: CmPopLoadReg + CmPopIncrSp, then back to CmIdle
  cm.popretz — 5'b11100: CmPopLoadReg + CmPopIncrSp + CmPopZeroA0 + CmPopRetRa
  cm.popret  — 5'b11110: CmPopLoadReg + CmPopIncrSp + CmPopRetRa
  cm.mvsa01  — casez 5'b011?? + instr[6:5]==01: CmMvSecondReg path (a0/a1 → s-regs)
  cm.mva01s  — casez 5'b011?? + instr[6:5]==11: CmMvSecondReg path (s-regs → a0/a1)

Encodinguri (16-bit, împachetate ca (C_NOP << 16) | rvc16, same convention as
codec_l9's _pack_rvc — câte o C_NOP urmează fiecare instrucțiune compressed):
  bits[15:13]=101 (funct3), bits[1:0]=10 (C2 quadrant)
  push/pop/popretz/popret: bits[12:8]=funct5, bits[7:4]=rlist, bits[3:2]=spimm
  mvsa01/mva01s:           bits[12]=0, bits[11:10]=11, bits[9:7]=r1s,
                           bits[6:5]=01/11, bits[4:2]=r2s

Parameterizare în action-space (rd/rs1/imm_bucket → rlist/spimm/r1s/r2s):
  rlist  = 4 + (rd % 12)   → 4-15, toate valorile legale
  spimm  = imm_bucket % 4  → 0-3
  r1s    = rd % 8
  r2s    = rs1 % 8

Total ops noi (Zcmp): 6  →  N_OPS = 143 + 6 = 149.
"""

import sys
from pathlib import Path

_L10 = Path(__file__).resolve().parent.parent / "level10_ops"
sys.path.insert(0, str(_L10))

from codec_l10 import (  # noqa: E402
    N_OPS as L10_N_OPS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    L10_CSRS,
    encode as l10_encode,
)

# CSR pool extins pentru max config:
#   + pmpcfg0-3  (0x3A0-0x3A3): configurare regiuni PMP — necesar pentru ibex_pmp bins
#   + pmpaddr0-15 (0x3B0-0x3BF): adrese regiuni PMP
#   + tselect/tdata1/tdata2 (0x7A0-0x7A2): debug triggers, legale doar cu DbgTriggerEn=1
#   + mhpmcounter/event 9-14 (0xB09-0xB0E, 0x329-0x32E): contoare HPM extinse
#   + cpuctrl (0x7C0): dummy_instr_en bit — necesar pentru ibex_dummy_instr coverage
#   + secureseed (0x7C1): dummy_instr_seed_o = csr_wdata_int, XOR-at in seed-ul LFSR
#     din ibex_dummy_instr.sv (dummy_instr_seed_d = seed_q ^ seed_i). Fara niciun
#     write pe acest CSR, LFSR-ul porneste mereu din RndCnstLfsrSeed (constanta de
#     compilare) si urmeaza aceeasi secventa determinista la fiecare episod — asta
#     bloca acoperirea ibex_dummy_instr la un plafon fix indiferent de antrenament.
L11_CSRS = L10_CSRS + [
    0x3A0, 0x3A1, 0x3A2, 0x3A3,                          # pmpcfg0..3
    0x3B0, 0x3B1, 0x3B2, 0x3B3, 0x3B4, 0x3B5, 0x3B6, 0x3B7,
    0x3B8, 0x3B9, 0x3BA, 0x3BB, 0x3BC, 0x3BD, 0x3BE, 0x3BF,  # pmpaddr0..15
    0x7A0, 0x7A1, 0x7A2,                                  # tselect, tdata1, tdata2
    0xB09, 0xB0A, 0xB0B, 0xB0C, 0xB0D, 0xB0E,            # mhpmcounter9..14
    0x329, 0x32A, 0x32B, 0x32C, 0x32D, 0x32E,            # mhpmevent9..14
    0x7C0,                                                 # cpuctrl: dummy_instr_en (SecureIbex)
    0x7C1,                                                 # secureseed: reseed LFSR dummy_instr
]
N_CSR_BUCKETS = len(L11_CSRS)

_CSR_F3  = {27: 0b001, 28: 0b010, 29: 0b011}   # CSRRW, CSRRS, CSRRC
_CSRI_F3 = {66: 0b101, 67: 0b110, 68: 0b111}   # CSRRWI, CSRRSI, CSRRCI


def _encode_csr_l11(funct3: int, rd: int, rs1: int, csr_bucket: int) -> int:
    csr = L11_CSRS[csr_bucket % len(L11_CSRS)]
    return (csr << 20) | ((rs1 & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _encode_csri_l11(funct3: int, rd: int, rs1: int, csr_bucket: int) -> int:
    uimm = rs1 & 0x1F
    csr  = L11_CSRS[csr_bucket % len(L11_CSRS)]
    return (csr << 20) | ((uimm & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011

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

# ── Op indices noi (116..142) — restul RV32BFull (draft legacy bitmanip,
#    identic cu RV32BOTEarlGrey pt. majoritatea): zbp/zbc/zbe/zbf rămase
#    neacoperite de Zba/Zbb/Zbs. Encodinguri verificate direct din
#    ibex_decoder.sv (blocurile illegal_insn pt. OPCODE_OP / OPCODE_OP_IMM).
SLO    = 116; SRO    = 117
GREV   = 118; GORC   = 119
SHFL   = 120; UNSHFL = 121
XPERM_N = 122; XPERM_B = 123; XPERM_H = 124
CLMUL  = 125; CLMULR = 126; CLMULH = 127
BCOMPRESS = 128; BDECOMPRESS = 129
BFP    = 130
SLOI   = 131; SROI   = 132
GREVI  = 133; GORCI  = 134
SHFLI  = 135; UNSHFLI = 136
CRC32_B = 137; CRC32_H = 138; CRC32_W = 139
CRC32C_B = 140; CRC32C_H = 141; CRC32C_W = 142

# ── Zcmp ops (143..148) — ibex_compressed_decoder.sv funct3=101 C2 quad ─────
CM_PUSH    = 143  # cm.push   — casez 5'b11000, CmPushStoreReg + CmPushDecrSp
CM_POP     = 144  # cm.pop    — casez 5'b11010, CmPopLoadReg + CmPopIncrSp
CM_POPRETZ = 145  # cm.popretz — casez 5'b11100, + CmPopZeroA0 + CmPopRetRa
CM_POPRET  = 146  # cm.popret  — casez 5'b11110, + CmPopRetRa
CM_MVSA01  = 147  # cm.mvsa01  — casez 5'b011?? + instr[6:5]==01, CmMvSecondReg
CM_MVA01S  = 148  # cm.mva01s  — casez 5'b011?? + instr[6:5]==11, CmMvSecondReg

N_OPS = 149

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
    # zbp/zbc/zbe/zbf (legacy draft bitmanip, RV32BOTEarlGrey|RV32BFull)
    SLO:    (0b001, 0b0010000), SRO:    (0b101, 0b0010000),
    GREV:   (0b101, 0b0110100), GORC:   (0b101, 0b0010100),
    SHFL:   (0b001, 0b0000100), UNSHFL: (0b101, 0b0000100),
    XPERM_N: (0b010, 0b0010100), XPERM_B: (0b100, 0b0010100), XPERM_H: (0b110, 0b0010100),
    CLMUL:  (0b001, 0b0000101), CLMULR: (0b010, 0b0000101), CLMULH: (0b011, 0b0000101),
    BCOMPRESS: (0b110, 0b0000100), BDECOMPRESS: (0b110, 0b0100100),
    BFP:    (0b111, 0b0100100),
}

# ── I-type shamt (opcode=0b0010011, funct7 fix + imm[4:0]=shamt) ───────────
_I_SHAMT_F3F7 = {
    RORI:  (0b101, 0b0110000),
    BCLRI: (0b001, 0b0100100),
    BSETI: (0b001, 0b0010100),
    BINVI: (0b001, 0b0110100),
    BEXTI: (0b101, 0b0100100),
    # zbp immediate variants (instr[25] e don't-care in decoder -> fixat pe 0)
    SLOI:  (0b001, 0b0010000),
    SROI:  (0b101, 0b0010000),
    GREVI: (0b101, 0b0110100),
    GORCI: (0b101, 0b0010100),
    SHFLI: (0b001, 0b0000100),
    UNSHFLI: (0b101, 0b0000100),
}

# ── I-type fixed imm12 (opcode=0b0010011, funct3=001, no shamt) ────────────
_I_FIXED12 = {
    CLZ:    0x600,
    CTZ:    0x601,
    CPOP:   0x602,
    SEXT_B: 0x604,
    SEXT_H: 0x605,
    # crc32/crc32c: imm[11:7]=01100 (grup CLZ), imm[6:5]=00, imm[4:0]=selector
    CRC32_B:  0x610,
    CRC32_H:  0x611,
    CRC32_W:  0x612,
    CRC32C_B: 0x618,
    CRC32C_H: 0x619,
    CRC32C_W: 0x61A,
}


_C_NOP = 0x0001  # c.nop = c.addi x0, 0 (same as codec_l9._C_NOP)


def _pack_rvc(rvc16: int) -> int:
    """Pack a 16-bit RVC instruction into a 32-bit word as (C_NOP << 16) | rvc16.
    Same convention as codec_l9._pack_rvc — each compressed slot is followed
    by one C_NOP so that fetch always sees a valid instruction in the upper half.
    """
    return (_C_NOP << 16) | (rvc16 & 0xFFFF)


def _zcmp_encode(op_i: int, rd: int, rs1: int, imm_bucket: int) -> int:
    """Encode one of the 6 Zcmp instructions (16-bit, packed via _pack_rvc).

    Bit layout (C2 quadrant bits[1:0]=10, funct3=101 bits[15:13]):
      cm.push/pop/popretz/popret: {101, funct5[4:0], rlist[3:0], spimm[1:0], 10}
        rlist  = 4 + (rd % 12)        → 4-15 (all legal rlist values)
        spimm  = imm_bucket % 4       → 0-3
      cm.mvsa01/mva01s: {101, 0, 11, r1s[2:0], mv_sel[1:0], r2s[2:0], 10}
        r1s    = rd % 8
        r2s    = rs1 % 8
        mv_sel = 01 (mvsa01) or 11 (mva01s)
    """
    base = (0b101 << 13) | 0b10  # funct3=101, C2 quadrant
    if op_i in (CM_PUSH, CM_POP, CM_POPRETZ, CM_POPRET):
        rlist = 4 + (rd % 12)
        spimm = imm_bucket % 4
        funct5 = {CM_PUSH: 0b11000, CM_POP: 0b11010,
                  CM_POPRETZ: 0b11100, CM_POPRET: 0b11110}[op_i]
        rvc16 = base | (funct5 << 8) | ((rlist & 0xF) << 4) | ((spimm & 0x3) << 2)
    elif op_i == CM_MVSA01:
        r1s = rd % 8
        r2s = rs1 % 8
        rvc16 = base | (0b011 << 10) | (r1s << 7) | (0b01 << 5) | (r2s << 2)
    else:  # CM_MVA01S
        r1s = rd % 8
        r2s = rs1 % 8
        rvc16 = base | (0b011 << 10) | (r1s << 7) | (0b11 << 5) | (r2s << 2)
    return _pack_rvc(rvc16)


def _r_type(f7: int, f3: int, rd: int, rs1: int, rs2: int) -> int:
    return (f7 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) \
         | (f3 << 12) | ((rd & 0x1F) << 7) | 0b0110011


def _i_shamt(f7: int, f3: int, rd: int, rs1: int, shamt: int) -> int:
    return (f7 << 25) | ((shamt & 0x1F) << 20) | ((rs1 & 0x1F) << 15) \
         | (f3 << 12) | ((rd & 0x1F) << 7) | 0b0010011


def _i_fixed(imm12: int, rd: int, rs1: int) -> int:
    return ((imm12 & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) \
         | (0b001 << 12) | ((rd & 0x1F) << 7) | 0b0010011


_ZCMP_OPS = {CM_PUSH, CM_POP, CM_POPRETZ, CM_POPRET, CM_MVSA01, CM_MVA01S}


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int, csr_bucket: int = 0) -> int:
    if op_i in _ZCMP_OPS:
        return _zcmp_encode(op_i, rd, rs1, imm_bucket)
    if op_i in _CSR_F3:
        return _encode_csr_l11(_CSR_F3[op_i], rd, rs1, csr_bucket)
    if op_i in _CSRI_F3:
        return _encode_csri_l11(_CSRI_F3[op_i], rd, rs1, csr_bucket)
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

    # Ops L10 (0..86) trebuie să rămână identice, EXCEPTÂND CSR ops care acum
    # folosesc L11_CSRS (mai mare) în loc de L10_CSRS.
    _csr_ops = set(_CSR_F3) | set(_CSRI_F3)
    for op_i in range(L10_N_OPS):
        if op_i in _csr_ops:
            continue
        for ib in range(IMM_BUCKETS):
            assert encode(op_i, 5, 6, 7, ib) == l10_encode(op_i, 5, 6, 7, ib), \
                f"L10 op={op_i} ib={ib} a fost modificat!"

    # Verifică că CSR ops folosesc L11_CSRS
    w = encode(27, 1, 2, 0, 0, 0)  # CSRRW cu csr_bucket=0 → L11_CSRS[0]
    assert ((w >> 20) & 0xFFF) == L11_CSRS[0], "CSRRW nu folosește L11_CSRS!"
    w = encode(27, 1, 2, 0, 0, len(L10_CSRS))  # primul CSR nou (pmpcfg0)
    assert ((w >> 20) & 0xFFF) == 0x3A0, f"pmpcfg0 encoding greșit: 0x{(w>>20)&0xFFF:03x}"

    # Zcmp: verifică structura bit-câmp pentru fiecare op
    # cm.push rlist=4 spimm=0: bits[15:13]=101, bits[12:8]=11000, bits[7:4]=0100, bits[3:2]=00, bits[1:0]=10
    w = encode(CM_PUSH, 0, 0, 0, 0)  # rd=0 → rlist=4+0=4, ib=0 → spimm=0
    rvc = w & 0xFFFF
    assert (rvc >> 13) == 0b101,    f"cm.push: funct3 != 101, rvc=0x{rvc:04x}"
    assert ((rvc >> 8) & 0x1F) == 0b11000, f"cm.push: funct5 != 11000"
    assert ((rvc >> 4) & 0xF) == 4, f"cm.push: rlist != 4"
    assert ((rvc >> 2) & 0x3) == 0, f"cm.push: spimm != 0"
    assert (rvc & 0x3) == 0b10,     f"cm.push: quad != C2"
    assert (w >> 16) == _C_NOP,     f"cm.push: upper half != C_NOP"
    # cm.pop rlist=6 spimm=1: rd=2 → rlist=4+2=6, ib=1 → spimm=1
    w = encode(CM_POP, 2, 0, 0, 1)
    rvc = w & 0xFFFF
    assert ((rvc >> 8) & 0x1F) == 0b11010, f"cm.pop: funct5 != 11010"
    assert ((rvc >> 4) & 0xF) == 6, f"cm.pop: rlist != 6"
    assert ((rvc >> 2) & 0x3) == 1, f"cm.pop: spimm != 1"
    # cm.mvsa01 r1s=3 r2s=5: rd=3 → r1s=3, rs1=5 → r2s=5
    w = encode(CM_MVSA01, 3, 5, 0, 0)
    rvc = w & 0xFFFF
    assert ((rvc >> 10) & 0x7) == 0b011, f"cm.mvsa01: bits[12:10] != 011"
    assert ((rvc >> 7) & 0x7) == 3,  f"cm.mvsa01: r1s != 3"
    assert ((rvc >> 5) & 0x3) == 0b01, f"cm.mvsa01: mv_sel != 01"
    assert ((rvc >> 2) & 0x7) == 5,  f"cm.mvsa01: r2s != 5"
    # cm.mva01s: mv_sel should be 11
    w = encode(CM_MVA01S, 3, 5, 0, 0)
    rvc = w & 0xFFFF
    assert ((rvc >> 5) & 0x3) == 0b11, f"cm.mva01s: mv_sel != 11"

    print(f"[OK] L11 codec self-test: {N_OPS} ops x {IMM_BUCKETS} buckets "
          f"(L10={L10_N_OPS} + {N_OPS - L10_N_OPS - 6} RV32B + 6 Zcmp)")
    print(f"  CSR pool: {N_CSR_BUCKETS} registre "
          f"(L10={len(L10_CSRS)}, +{N_CSR_BUCKETS - len(L10_CSRS)} max-specific)")


if __name__ == "__main__":
    _self_test()
