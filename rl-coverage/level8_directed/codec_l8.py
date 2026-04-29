"""Level 8 codec — L7 (64 ops) + LUI + JALR + CSRRWI/SI/CI + FENCE = 70 ops.

These 6 additions are the biggest missing instruction types for LINE coverage:
  + LUI   (U-type)    — ibex_tracer + ibex_decoder LUI branch
  + JALR  (I-type)    — ibex_tracer JALR branch + indirect-call path in controller
  + CSRRWI/CSRRSI/CSRRCI (immediate CSR ops) — 3 more tracer branches
  + FENCE (MISC-MEM)  — ibex_decoder/controller MISC-MEM branch

Each missing op corresponds to at least one tracer case arm that was never hit,
plus associated decoder/ALU/controller paths.  Adding them here lets both the
directed program (directed_l8.py) and the PPO agent emit them.

Immediate buckets for JALR: we compute a relative offset from the current PC.
To keep it simple for the agent we offer 5 offset buckets (in bytes) that are
all small-positive, meaning JALR typically lands a few instructions ahead —
safe as long as we set rs1 correctly via a JAL or AUIPC first.
"""

import sys
from pathlib import Path
from enum import IntEnum

_L7 = (Path(__file__).resolve().parent.parent / "level7_stimulus")
sys.path.insert(0, str(_L7))

from codec_l7 import (  # noqa: E402
    Op as L7Op,
    N_OPS as L7_N_OPS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    SHAMT_BUCKET_VALUES,
    BRANCH_BUCKET_OFFSETS,
    JAL_BUCKET_OFFSETS,
    L7_SAFE_CSRS,
    encode as l7_encode,
    emit_program as _l7_emit,
)


class Op(IntEnum):
    # ── Keep L7 ops unchanged (0-63) ──────────────────────────────────────
    ADD = 0;   SUB = 1;   SLL = 2;   SLT = 3;   SLTU = 4
    XOR = 5;   SRL = 6;   SRA = 7;   OR = 8;    AND = 9
    ADDI = 10; SLTI = 11; SLTIU = 12; XORI = 13; ORI = 14; ANDI = 15
    SLLI = 16; SRLI = 17; SRAI = 18
    LB = 19;   LH = 20;   LW = 21;   LBU = 22;  LHU = 23
    SB = 24;   SH = 25;   SW = 26
    CSRRW = 27; CSRRS = 28; CSRRC = 29
    MUL = 30; MULH = 31; MULHSU = 32; MULHU = 33
    DIV = 34; DIVU = 35; REM = 36; REMU = 37
    BEQ = 38; BNE = 39; BLT = 40; BGE = 41; BLTU = 42; BGEU = 43
    JAL = 44
    C_ADDI = 45; C_LI = 46; C_LUI = 47; C_SLLI = 48
    C_MV = 49;   C_ADD = 50
    C_ADDI4SPN = 51; C_LW = 52; C_SW = 53
    C_AND = 54; C_OR = 55; C_XOR = 56; C_SUB = 57
    C_SRLI = 58; C_SRAI = 59; C_ANDI = 60
    AUIPC  = 61
    ECALL  = 62
    EBREAK = 63
    # ── Level 8 additions ─────────────────────────────────────────────────
    LUI    = 64   # U-type, same 20-bit upper-imm buckets as AUIPC
    JALR   = 65   # I-type: jump-and-link-register
    CSRRWI = 66   # CSR immediate write  (funct3=0b101)
    CSRRSI = 67   # CSR immediate set    (funct3=0b110)
    CSRRCI = 68   # CSR immediate clear  (funct3=0b111)
    FENCE  = 69   # Memory fence (MISC-MEM opcode)


N_OPS = 70

# Re-use AUIPC's IMM buckets for LUI (same upper-20 immediate space)
LUI_IMM_BUCKETS = [0x00001, 0x12345, 0xABCDE, 0xFFFFF, 0x80000]

# JALR offset buckets (byte offsets).  We intend to call via a link register
# set by a preceding JAL, so JALR(rs1=link_reg, offset) = link_reg + offset.
# We use small offsets ±0 so a "JALR x0, 0(x1)" is a simple return.
JALR_OFFSET_BUCKETS = [0, 4, 8, -4, 0]   # 5 buckets, mirroring IMM_BUCKETS

# CSR immediate uimm[4:0] buckets (unsigned 5-bit)
CSRIMM_BUCKETS = [0, 1, 3, 7, 15]


def _encode_lui(rd: int, imm_bucket: int) -> int:
    """LUI rd, imm  — rd = imm << 12. U-type, opcode=0b0110111."""
    imm20 = LUI_IMM_BUCKETS[imm_bucket % len(LUI_IMM_BUCKETS)] & 0xFFFFF
    return (imm20 << 12) | ((rd & 0x1F) << 7) | 0b0110111


def _encode_jalr(rd: int, rs1: int, imm_bucket: int) -> int:
    """JALR rd, rs1, offset  — rd=PC+4, PC=rs1+offset. I-type, opcode=0b1100111."""
    offset = JALR_OFFSET_BUCKETS[imm_bucket % len(JALR_OFFSET_BUCKETS)] & 0xFFF
    return (offset << 20) | ((rs1 & 0x1F) << 15) | (0b000 << 12) | ((rd & 0x1F) << 7) | 0b1100111


def _encode_csrimm(funct3: int, rd: int, uimm5: int, imm_bucket: int) -> int:
    """CSRRWI/CSRRSI/CSRRCI with 5-bit unsigned immediate."""
    uimm = CSRIMM_BUCKETS[imm_bucket % len(CSRIMM_BUCKETS)] & 0x1F
    # Select CSR address using same scheme as L7 CSR encoding
    csr_idx = (imm_bucket * 7 + uimm) % len(L7_SAFE_CSRS)
    csr = L7_SAFE_CSRS[csr_idx]
    return (csr << 20) | ((uimm & 0x1F) << 15) | (funct3 << 12) | ((rd & 0x1F) << 7) | 0b1110011


def _encode_fence() -> int:
    """Simple FENCE instruction: predecessor=IORW, successor=IORW."""
    # FENCE iorw,iorw = 0x0FF0000F
    return 0x0FF0000F


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    """Encode action tuple → 32-bit RISC-V instruction word."""
    if op_i == int(Op.LUI):
        return _encode_lui(rd, imm_bucket)
    if op_i == int(Op.JALR):
        return _encode_jalr(rd, rs1, imm_bucket)
    if op_i == int(Op.CSRRWI):
        return _encode_csrimm(0b101, rd, rs1, imm_bucket)
    if op_i == int(Op.CSRRSI):
        return _encode_csrimm(0b110, rd, rs1, imm_bucket)
    if op_i == int(Op.CSRRCI):
        return _encode_csrimm(0b111, rd, rs1, imm_bucket)
    if op_i == int(Op.FENCE):
        return _encode_fence()
    # All other ops: delegate to L7
    return l7_encode(op_i, rd, rs1, rs2, imm_bucket)


def emit_program(actions):
    """One 32-bit word per action + 16 NOP tail (same contract as L7)."""
    nop = encode(int(Op.ADDI), 0, 0, 0, 2)  # ADDI x0, x0, 0
    return [encode(*a) for a in actions] + [nop] * 16


def _self_test():
    for op_i in range(N_OPS):
        for ib in range(IMM_BUCKETS):
            w = encode(op_i, 5, 6, 7, ib)
            assert 0 <= w <= 0xFFFFFFFF, f"op={op_i} ib={ib} → 0x{w:08x}"
    assert (encode(int(Op.LUI), 3, 0, 0, 0) & 0x7F) == 0b0110111
    assert (encode(int(Op.JALR), 1, 2, 0, 0) & 0x7F) == 0b1100111
    assert encode(int(Op.FENCE), 0, 0, 0, 0) == 0x0FF0000F
    print(f"[OK] L8 self-test: {N_OPS} ops × {IMM_BUCKETS} buckets")
    print(f"  LUI   sample: 0x{encode(int(Op.LUI),  3,0,0,1):08x}")
    print(f"  JALR  sample: 0x{encode(int(Op.JALR), 1,2,0,0):08x}")
    print(f"  CSRRWI sample: 0x{encode(int(Op.CSRRWI),1,1,0,0):08x}")
    print(f"  FENCE: 0x{encode(int(Op.FENCE),0,0,0,0):08x}")


if __name__ == "__main__":
    _self_test()
