"""coverage_model.py — Rich Functional Coverage Model pentru Ibex (~7000 bins).

Bin-uri:
  1. SEEN          (63)    — fiecare instrucțiune executată
  2. OPERAND_PAIR  (~2268) — combinație rs1_cat × rs2_cat per instrucțiune
  3. RD_ZERO       (~30)   — instrucțiuni cu rd = x0
  4. RS_ZERO       (~40)   — instrucțiuni cu rs1=x0 sau rs2=x0
  5. SAME_SRC      (~20)   — instrucțiuni cu rs1_addr == rs2_addr
  6. IMM_RANGE     (~180)  — categoria imediatului pentru I-type/branch
  7. RAW_HAZARD    (~3000) — perechi producer→consumer la distance 1,2,3
  8. SEQUENCE      (~1500) — perechi consecutive de instrucțiuni
  9. CORNER_CASES  (~50)   — divide/0, misaligned, branch extremes, overflow
"""

from __future__ import annotations
from enum import IntEnum
from typing import Optional
import json


# ── Categorii operanzi ────────────────────────────────────────────────────

class OpCat(IntEnum):
    ZERO      = 0
    SMALL_POS = 1   # 1..15
    LARGE_POS = 2   # 16..2047
    SMALL_NEG = 3   # -1..-15
    LARGE_NEG = 4   # -16..-2048
    EXTREME   = 5   # INT_MIN, INT_MAX, 0xFFFFFFFF, >2047 sau <-2048

N_OP_CATS = 6

def categorize(val32: int) -> int:
    v = val32 & 0xFFFFFFFF
    signed = v if v < 0x80000000 else v - 0x100000000
    if signed == 0: return 0
    if v == 0xFFFFFFFF or signed == 0x7FFFFFFF or signed == -0x80000000: return 5
    if 1 <= signed <= 15:    return 1
    if 16 <= signed <= 2047: return 2
    if -15 <= signed <= -1:  return 3
    if -2048 <= signed <= -16: return 4
    return 5


# ── Instrucțiuni (63 tipuri) ──────────────────────────────────────────────

INSTR_TYPES = [
    # (name, opcode, funct3_or_None, funct7_or_f12_or_None)
    ("ADD",    0x33, 0x0, 0x00), ("SUB",    0x33, 0x0, 0x20),
    ("SLL",    0x33, 0x1, 0x00), ("SLT",    0x33, 0x2, 0x00),
    ("SLTU",   0x33, 0x3, 0x00), ("XOR",    0x33, 0x4, 0x00),
    ("SRL",    0x33, 0x5, 0x00), ("SRA",    0x33, 0x5, 0x20),
    ("OR",     0x33, 0x6, 0x00), ("AND",    0x33, 0x7, 0x00),
    ("MUL",    0x33, 0x0, 0x01), ("MULH",   0x33, 0x1, 0x01),
    ("MULHSU", 0x33, 0x2, 0x01), ("MULHU",  0x33, 0x3, 0x01),
    ("DIV",    0x33, 0x4, 0x01), ("DIVU",   0x33, 0x5, 0x01),
    ("REM",    0x33, 0x6, 0x01), ("REMU",   0x33, 0x7, 0x01),
    ("ADDI",   0x13, 0x0, None), ("SLTI",   0x13, 0x2, None),
    ("SLTIU",  0x13, 0x3, None), ("XORI",   0x13, 0x4, None),
    ("ORI",    0x13, 0x6, None), ("ANDI",   0x13, 0x7, None),
    ("SLLI",   0x13, 0x1, 0x00), ("SRLI",   0x13, 0x5, 0x00),
    ("SRAI",   0x13, 0x5, 0x20),
    ("LB",     0x03, 0x0, None), ("LH",     0x03, 0x1, None),
    ("LW",     0x03, 0x2, None), ("LBU",    0x03, 0x4, None),
    ("LHU",    0x03, 0x5, None),
    ("SB",     0x23, 0x0, None), ("SH",     0x23, 0x1, None),
    ("SW",     0x23, 0x2, None),
    ("BEQ",    0x63, 0x0, None), ("BNE",    0x63, 0x1, None),
    ("BLT",    0x63, 0x4, None), ("BGE",    0x63, 0x5, None),
    ("BLTU",   0x63, 0x6, None), ("BGEU",   0x63, 0x7, None),
    ("JAL",    0x6F, None, None), ("JALR",  0x67, 0x0, None),
    ("LUI",    0x37, None, None), ("AUIPC", 0x17, None, None),
    ("ECALL",  0x73, 0x0, 0x000), ("EBREAK",0x73, 0x0, 0x001),
    ("MRET",   0x73, 0x0, 0x302), ("WFI",   0x73, 0x0, 0x105),
    ("CSRRW",  0x73, 0x1, None), ("CSRRS",  0x73, 0x2, None),
    ("CSRRC",  0x73, 0x3, None), ("CSRRWI", 0x73, 0x5, None),
    ("CSRRSI", 0x73, 0x6, None), ("CSRRCI", 0x73, 0x7, None),
    ("FENCE",  0x0F, 0x0, None), ("FENCEI", 0x0F, 0x1, None),
    ("C_ADDI", 0xFF, None, None), ("C_LI",  0xFE, None, None),
    ("C_LW",   0xFD, None, None), ("C_SW",  0xFC, None, None),
    ("C_ADD",  0xFB, None, None), ("C_MV",  0xFA, None, None),
    ("C_BEQZ", 0xF9, None, None), ("C_BNEZ",0xF8, None, None),
    ("C_J",    0xF7, None, None), ("C_SLLI",0xF6, None, None),
]

N_INSTRS    = len(INSTR_TYPES)
INSTR_NAMES = [t[0] for t in INSTR_TYPES]
INSTR_INDEX = {n: i for i, n in enumerate(INSTR_NAMES)}

HAS_RD  = set(INSTR_NAMES) - {"SB","SH","SW","BEQ","BNE","BLT","BGE","BLTU","BGEU",
                                "ECALL","EBREAK","MRET","WFI","FENCE","FENCEI",
                                "C_SW","C_BEQZ","C_BNEZ","C_J"}
HAS_RS1 = set(INSTR_NAMES) - {"LUI","AUIPC","JAL","ECALL","EBREAK","MRET","WFI",
                                "FENCE","FENCEI","C_J"}
HAS_RS2 = {"ADD","SUB","SLL","SLT","SLTU","XOR","SRL","SRA","OR","AND",
            "MUL","MULH","MULHSU","MULHU","DIV","DIVU","REM","REMU",
            "SB","SH","SW","BEQ","BNE","BLT","BGE","BLTU","BGEU",
            "C_ADD","C_MV","C_SW"}
HAS_IMM = {"ADDI","SLTI","SLTIU","XORI","ORI","ANDI","SLLI","SRLI","SRAI",
            "LB","LH","LW","LBU","LHU","SB","SH","SW",
            "BEQ","BNE","BLT","BGE","BLTU","BGEU","JAL","JALR","LUI","AUIPC",
            "CSRRWI","CSRRSI","CSRRCI"}


def decode_instr(insn: int) -> Optional[int]:
    insn &= 0xFFFFFFFF
    if (insn & 0x3) != 0x3:
        hw = insn & 0xFFFF
        q  = hw & 0x3
        f3 = (hw >> 13) & 0x7
        if q == 0b01:
            if f3 == 0: return INSTR_INDEX.get("C_ADDI")
            if f3 == 2: return INSTR_INDEX.get("C_LI")
            if f3 == 6: return INSTR_INDEX.get("C_BEQZ")
            if f3 == 7: return INSTR_INDEX.get("C_BNEZ")
            if f3 in (1,5): return INSTR_INDEX.get("C_J")
        if q == 0b00:
            if f3 == 2: return INSTR_INDEX.get("C_LW")
            if f3 == 6: return INSTR_INDEX.get("C_SW")
        if q == 0b10:
            if f3 == 0: return INSTR_INDEX.get("C_SLLI")
            if f3 == 4:
                rs2  = (hw >> 2) & 0x1F
                b12  = (hw >> 12) & 1
                if b12 == 0 and rs2 != 0: return INSTR_INDEX.get("C_MV")
                if b12 == 1 and rs2 != 0: return INSTR_INDEX.get("C_ADD")
        return None

    op  = insn & 0x7F
    f3  = (insn >> 12) & 0x7
    f7  = (insn >> 25) & 0x7F
    f12 = (insn >> 20) & 0xFFF

    for idx, (name, iop, tf3, tf7) in enumerate(INSTR_TYPES):
        if iop > 0xF0: continue
        if iop != op:  continue
        if tf3 is not None and tf3 != f3: continue
        if tf7 is not None:
            if op == 0x73 and f3 == 0:
                if tf7 != f12: continue
            elif tf7 != f7:
                continue
        return idx
    return None


def extract_imm(insn: int) -> Optional[int]:
    op = insn & 0x7F
    if op in (0x13, 0x03, 0x67):
        imm = (insn >> 20) & 0xFFF
        if imm & 0x800: imm |= ~0xFFF
        return imm
    if op == 0x23:
        imm = (((insn>>25)&0x7F)<<5) | ((insn>>7)&0x1F)
        if imm & 0x800: imm |= ~0xFFF
        return imm
    if op == 0x63:
        imm = (((insn>>31)&1)<<12)|(((insn>>7)&1)<<11)|\
              (((insn>>25)&0x3F)<<5)|(((insn>>8)&0xF)<<1)
        if imm & 0x1000: imm |= ~0x1FFF
        return imm
    if op in (0x37, 0x17): return (insn>>12) & 0xFFFFF
    if op == 0x6F:
        imm = (((insn>>31)&1)<<20)|(((insn>>12)&0xFF)<<12)|\
              (((insn>>20)&1)<<11)|(((insn>>21)&0x3FF)<<1)
        if imm & 0x100000: imm |= ~0x1FFFFF
        return imm
    return None


class CoverageModel:
    def __init__(self):
        self.seen         = [False] * N_INSTRS
        self.operand_pair = [[[False]*N_OP_CATS for _ in range(N_OP_CATS)]
                              for _ in range(N_INSTRS)]
        self.rd_zero      = [False] * N_INSTRS
        self.rs_zero      = [[False, False] for _ in range(N_INSTRS)]
        self.same_src     = [False] * N_INSTRS
        self.imm_range    = [[False]*N_OP_CATS for _ in range(N_INSTRS)]
        self.raw_hazard   = [[[False]*N_INSTRS for _ in range(N_INSTRS)]
                              for _ in range(3)]
        self.sequence     = [[False]*N_INSTRS for _ in range(N_INSTRS)]
        self.corner       = self._make_corner_bins()
        self._history: list[tuple[int,int]] = []
        self._prev_idx: Optional[int] = None

    @staticmethod
    def _make_corner_bins() -> dict:
        d = {}
        for op in ("DIV","DIVU","REM","REMU"):
            d[f"div_zero_{op}"] = False
        for op in ("LH","LHU"):
            for off in (1,3): d[f"misaligned_{op}_off{off}"] = False
        for off in (1,2,3):  d[f"misaligned_LW_off{off}"]   = False
        for op in ("SH",):
            for off in (1,3): d[f"misaligned_{op}_off{off}"] = False
        for off in (1,2,3):  d[f"misaligned_SW_off{off}"]   = False
        for op in ("BEQ","BNE","BLT","BGE","BLTU","BGEU"):
            d[f"branch_taken_{op}"]    = False
            d[f"branch_ntaken_{op}"]   = False
            d[f"branch_backward_{op}"] = False
        d["mul_overflow_INT_MAX"]  = False
        d["addi_overflow_INT_MAX"] = False
        for op in ("SLL","SRL","SRA","SLLI","SRLI","SRAI"):
            d[f"shift_by_0_{op}"]  = False
            d[f"shift_by_31_{op}"] = False
        d["ecall_trap"]        = False
        d["ebreak_trap"]       = False
        d["jalr_rd0"]          = False
        d["jalr_rd_link"]      = False
        d["add_int_max_plus_1"]  = False
        d["sub_int_min_minus_1"] = False
        d["xor_all_ones"]      = False
        d["and_all_zeros"]     = False
        return d

    def sample(self, rv: dict):
        insn      = rv["insn"]      & 0xFFFFFFFF
        rs1_addr  = rv["rs1_addr"]  & 0x1F
        rs2_addr  = rv["rs2_addr"]  & 0x1F
        rd_addr   = rv["rd_addr"]   & 0x1F
        rs1_rdata = rv["rs1_rdata"] & 0xFFFFFFFF
        rs2_rdata = rv["rs2_rdata"] & 0xFFFFFFFF
        rd_wdata  = rv["rd_wdata"]  & 0xFFFFFFFF
        pc_rdata  = rv["pc_rdata"]  & 0xFFFFFFFF
        pc_wdata  = rv["pc_wdata"]  & 0xFFFFFFFF
        mem_addr  = rv.get("mem_addr", 0) & 0xFFFFFFFF
        trap      = bool(rv.get("trap", False))

        idx = decode_instr(insn)
        if idx is None:
            self._prev_idx = None
            self._history.clear()
            return
        name = INSTR_NAMES[idx]

        # 1. SEEN
        self.seen[idx] = True

        # 2. OPERAND_PAIR
        r1c = categorize(rs1_rdata)
        r2c = categorize(rs2_rdata)
        self.operand_pair[idx][r1c][r2c] = True

        # 3. RD_ZERO
        if name in HAS_RD and rd_addr == 0:
            self.rd_zero[idx] = True

        # 4. RS_ZERO
        if name in HAS_RS1 and rs1_addr == 0: self.rs_zero[idx][0] = True
        if name in HAS_RS2 and rs2_addr == 0: self.rs_zero[idx][1] = True

        # 5. SAME_SRC
        if name in HAS_RS2 and rs1_addr == rs2_addr and rs1_addr != 0:
            self.same_src[idx] = True

        # 6. IMM_RANGE
        if name in HAS_IMM:
            imm = extract_imm(insn)
            if imm is not None:
                self.imm_range[idx][categorize(imm & 0xFFFFFFFF)] = True

        # 7. RAW_HAZARD
        self._history.append((idx, rd_addr))
        if len(self._history) > 4: self._history.pop(0)
        if name in HAS_RS1 or name in HAS_RS2:
            reads = set()
            if name in HAS_RS1 and rs1_addr != 0: reads.add(rs1_addr)
            if name in HAS_RS2 and rs2_addr != 0: reads.add(rs2_addr)
            for dist in range(1, 4):
                pos = len(self._history) - 1 - dist
                if pos >= 0:
                    pi, prd = self._history[pos]
                    if prd != 0 and prd in reads and INSTR_NAMES[pi] in HAS_RD:
                        self.raw_hazard[dist-1][pi][idx] = True

        # 8. SEQUENCE
        if self._prev_idx is not None:
            self.sequence[self._prev_idx][idx] = True
        self._prev_idx = idx

        # 9. CORNERS
        self._corners(name, insn, rs1_addr, rs2_addr, rd_addr,
                      rs1_rdata, rs2_rdata, rd_wdata,
                      pc_rdata, pc_wdata, mem_addr, trap)

    def _corners(self, name, insn, rs1_addr, rs2_addr, rd_addr,
                 rs1_rdata, rs2_rdata, rd_wdata,
                 pc_rdata, pc_wdata, mem_addr, trap):
        c = self.corner
        if name in ("DIV","DIVU","REM","REMU") and rs2_rdata == 0:
            c[f"div_zero_{name}"] = True
        if name in ("LH","LHU"):
            off = mem_addr & 1
            if off: c[f"misaligned_{name}_off1"] = c[f"misaligned_{name}_off3"] = True
        if name == "LW":
            off = mem_addr & 3
            if off in (1,2,3): c[f"misaligned_LW_off{off}"] = True
        if name == "SH":
            if mem_addr & 1: c["misaligned_SH_off1"] = c["misaligned_SH_off3"] = True
        if name == "SW":
            off = mem_addr & 3
            if off in (1,2,3): c[f"misaligned_SW_off{off}"] = True
        if name in ("BEQ","BNE","BLT","BGE","BLTU","BGEU"):
            taken = pc_wdata != pc_rdata + 4
            if taken:
                c[f"branch_taken_{name}"] = True
                if pc_wdata < pc_rdata: c[f"branch_backward_{name}"] = True
            else:
                c[f"branch_ntaken_{name}"] = True
        if name == "MUL" and rs1_rdata == 0x7FFFFFFF and rs2_rdata == 0x7FFFFFFF:
            c["mul_overflow_INT_MAX"] = True
        if name == "ADDI":
            imm = extract_imm(insn)
            if imm == 1 and rs1_rdata == 0x7FFFFFFF:
                c["addi_overflow_INT_MAX"] = True
        for op in ("SLL","SRL","SRA"):
            if name == op:
                sh = rs2_rdata & 0x1F
                if sh == 0:  c[f"shift_by_0_{op}"]  = True
                if sh == 31: c[f"shift_by_31_{op}"] = True
        for op in ("SLLI","SRLI","SRAI"):
            if name == op:
                sh = (insn >> 20) & 0x1F
                if sh == 0:  c[f"shift_by_0_{op}"]  = True
                if sh == 31: c[f"shift_by_31_{op}"] = True
        if name == "ECALL"  and trap: c["ecall_trap"]  = True
        if name == "EBREAK" and trap: c["ebreak_trap"] = True
        if name == "JALR":
            if rd_addr == 0: c["jalr_rd0"]     = True
            else:            c["jalr_rd_link"]  = True
        if name == "ADD" and (rs1_rdata == 0x7FFFFFFF or rs2_rdata == 0x7FFFFFFF):
            c["add_int_max_plus_1"] = True
        if name == "SUB" and (rs1_rdata == 0x80000000 or rs2_rdata == 0x80000000):
            c["sub_int_min_minus_1"] = True
        if name == "XOR" and rd_wdata == 0xFFFFFFFF: c["xor_all_ones"]  = True
        if name == "AND" and rd_wdata == 0:          c["and_all_zeros"] = True

    def hit_set(self) -> set:
        hits = set()
        for i,n in enumerate(INSTR_NAMES):
            if self.seen[i]: hits.add(f"seen_{n}")
            if n in HAS_RD and self.rd_zero[i]:   hits.add(f"rdzero_{n}")
            if n in HAS_RS1 and self.rs_zero[i][0]: hits.add(f"rs1zero_{n}")
            if n in HAS_RS2 and self.rs_zero[i][1]: hits.add(f"rs2zero_{n}")
            if n in HAS_RS2 and self.same_src[i]: hits.add(f"samesrc_{n}")
            if n in HAS_RS1 or n in HAS_RS2:
                for r1 in range(N_OP_CATS):
                    for r2 in range(N_OP_CATS):
                        if self.operand_pair[i][r1][r2]:
                            hits.add(f"opair_{n}_{r1}_{r2}")
            if n in HAS_IMM:
                for cat in range(N_OP_CATS):
                    if self.imm_range[i][cat]: hits.add(f"imm_{n}_{cat}")
        for d in range(3):
            for pi,pn in enumerate(INSTR_NAMES):
                for ci,cn in enumerate(INSTR_NAMES):
                    if self.raw_hazard[d][pi][ci]: hits.add(f"raw{d+1}_{pn}_{cn}")
        for pi,pn in enumerate(INSTR_NAMES):
            for ci,cn in enumerate(INSTR_NAMES):
                if self.sequence[pi][ci]: hits.add(f"seq_{pn}_{cn}")
        for k,v in self.corner.items():
            if v: hits.add(f"corner_{k}")
        return hits

    def count_bins(self) -> tuple:
        covered = 0; total = 0
        covered += sum(self.seen);           total += N_INSTRS
        for i,n in enumerate(INSTR_NAMES):
            if n in HAS_RD: covered += self.rd_zero[i]; total += 1
            if n in HAS_RS1: covered += self.rs_zero[i][0]; total += 1
            if n in HAS_RS2: covered += self.rs_zero[i][1]; total += 1
            if n in HAS_RS2: covered += self.same_src[i]; total += 1
            if n in HAS_RS1 or n in HAS_RS2:
                for r1 in range(N_OP_CATS):
                    for r2 in range(N_OP_CATS):
                        covered += self.operand_pair[i][r1][r2]; total += 1
            if n in HAS_IMM:
                for cat in range(N_OP_CATS):
                    covered += self.imm_range[i][cat]; total += 1
        for d in range(3):
            for pi,pn in enumerate(INSTR_NAMES):
                if pn not in HAS_RD: continue
                for ci,cn in enumerate(INSTR_NAMES):
                    if cn not in HAS_RS1 and cn not in HAS_RS2: continue
                    covered += self.raw_hazard[d][pi][ci]; total += 1
        for pi in range(N_INSTRS):
            for ci in range(N_INSTRS):
                covered += self.sequence[pi][ci]; total += 1
        covered += sum(self.corner.values()); total += len(self.corner)
        return covered, total

    def print_report(self):
        cov, tot = self.count_bins()
        print(f"\n{'='*50}")
        print(f"RICH FUNCTIONAL COVERAGE")
        print(f"  TOTAL: {cov}/{tot} = {100*cov/max(tot,1):.2f}%")
        print(f"{'='*50}")
        seen_c = sum(self.seen)
        print(f"  seen:     {seen_c}/{N_INSTRS}")
        corner_c = sum(self.corner.values())
        print(f"  corners:  {corner_c}/{len(self.corner)}")
        for d in range(3):
            rc = sum(self.raw_hazard[d][p][c]
                     for p in range(N_INSTRS) for c in range(N_INSTRS))
            rt = sum(1 for pn in INSTR_NAMES if pn in HAS_RD
                     for cn in INSTR_NAMES if cn in HAS_RS1 or cn in HAS_RS2)
            print(f"  raw_d{d+1}:   {rc}/{rt}")
        sc = sum(self.sequence[p][c] for p in range(N_INSTRS) for c in range(N_INSTRS))
        print(f"  sequence: {sc}/{N_INSTRS*N_INSTRS}")
        print(f"{'='*50}")
