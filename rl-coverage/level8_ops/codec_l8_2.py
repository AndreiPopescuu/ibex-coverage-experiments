"""Level 8.2 codec — L8 (70 ops) + MRET + WFI = 72 ops.

MRET (0x30200073): return din trap handler.
  Exercită ibex_controller FSM path-ul de MRET, pc_mux_o=PC_EXC,
  mstatus.mie restore, mepc→PC. Fără asta, jumătate din FSM-ul
  de excepții rămâne neacoperit indiferent de câte ECALL faci.

WFI (0x10500073): wait for interrupt.
  Exercită calea WFI din ibex_controller și ibex_cs_registers.
"""

import sys
from pathlib import Path
from enum import IntEnum

_L8 = Path(__file__).resolve().parent
sys.path.insert(0, str(_L8))

from codec_l8 import (  # noqa: E402
    Op as L8Op,
    N_OPS as L8_N_OPS,
    IMM_BUCKETS,
    IMM_BUCKET_VALUES,
    encode as l8_encode,
)


class Op(IntEnum):
    # L8 ops unchanged (0-69)
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
    LUI    = 64
    JALR   = 65
    CSRRWI = 66
    CSRRSI = 67
    CSRRCI = 68
    FENCE  = 69
    # L8.2 additions
    MRET   = 70
    WFI    = 71


N_OPS = 72

_MRET = 0x30200073
_WFI  = 0x10500073


def encode(op_i: int, rd: int, rs1: int, rs2: int, imm_bucket: int) -> int:
    if op_i == int(Op.MRET):
        return _MRET
    if op_i == int(Op.WFI):
        return _WFI
    return l8_encode(op_i, rd, rs1, rs2, imm_bucket)


def emit_program(actions):
    nop = encode(int(Op.ADDI), 0, 0, 0, 2)
    return [encode(*a) for a in actions] + [nop] * 16


def _self_test():
    for op_i in range(N_OPS):
        for ib in range(IMM_BUCKETS):
            w = encode(op_i, 5, 6, 7, ib)
            assert 0 <= w <= 0xFFFFFFFF, f"op={op_i} ib={ib} → 0x{w:08x}"
    assert encode(int(Op.MRET), 0, 0, 0, 0) == _MRET
    assert encode(int(Op.WFI),  0, 0, 0, 0) == _WFI
    print(f"[OK] L8.2 self-test: {N_OPS} ops × {IMM_BUCKETS} buckets")
    print(f"  MRET: 0x{_MRET:08x}")
    print(f"  WFI:  0x{_WFI:08x}")


if __name__ == "__main__":
    _self_test()
