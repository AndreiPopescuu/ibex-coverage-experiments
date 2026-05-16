"""shadow_riscv.py — Python RISC-V ISA simulator matching Ibex's test_run_for_l9 prologue.

Simulates execution of the 70-op codec (level8/9) without Verilator.
Returns the same trace dict that CoverageModel.sample() expects.

Memory model matches DiverseMemAgent: unknown addresses read as addr ^ 0xDEADBEEF.
Prologue register state matches _build_prologue() in test_run_for_l9.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from coverage_model import CoverageModel
from codec_l8 import emit_program

# Trap handler exact as in test_run_for_l9.py
# mtvec = 0x00200000, handler:
#   CSRRS x10, mepc, x0  → x10 = mepc (faulting PC)
#   ADDI  x10, x10, 4    → x10 = faulting PC + 4
#   CSRRW x0,  mepc, x10 → mepc = faulting PC + 4
#   MRET                  → PC = mepc
_TRAP_BASE   = 0x00200000
_CSRRS_X10_MEPC_X0 = 0x34102573
_ADDI_X10_X10_4    = 0x00450513
_CSRRW_X0_MEPC_X10 = 0x34151073
_MRET              = 0x30200073
_WFI               = 0x10500073


def _trap_handler_traces(trap_pc: int) -> list:
    """Synthetic traces for the 4-instruction trap handler."""
    pc4 = (trap_pc + 4) & 0xFFFFFFFF
    return [
        {"insn": _CSRRS_X10_MEPC_X0, "rs1_addr": 0,  "rs2_addr": 0, "rd_addr": 10,
         "rs1_rdata": 0,       "rs2_rdata": 0, "rd_wdata": trap_pc,
         "pc_rdata": _TRAP_BASE,    "pc_wdata": _TRAP_BASE+4, "mem_addr": 0, "trap": False},
        {"insn": _ADDI_X10_X10_4,    "rs1_addr": 10, "rs2_addr": 0, "rd_addr": 10,
         "rs1_rdata": trap_pc, "rs2_rdata": 0, "rd_wdata": pc4,
         "pc_rdata": _TRAP_BASE+4,  "pc_wdata": _TRAP_BASE+8, "mem_addr": 0, "trap": False},
        {"insn": _CSRRW_X0_MEPC_X10, "rs1_addr": 10, "rs2_addr": 0, "rd_addr": 0,
         "rs1_rdata": pc4,     "rs2_rdata": 0, "rd_wdata": trap_pc,
         "pc_rdata": _TRAP_BASE+8,  "pc_wdata": _TRAP_BASE+12,"mem_addr": 0, "trap": False},
        {"insn": _MRET,              "rs1_addr": 0,  "rs2_addr": 0, "rd_addr": 0,
         "rs1_rdata": 0,       "rs2_rdata": 0, "rd_wdata": 0,
         "pc_rdata": _TRAP_BASE+12, "pc_wdata": pc4,           "mem_addr": 0, "trap": False},
    ]

# ── Prologue register state (from test_run_for_l9._build_prologue) ────────────
# x1-x15  = 1..15  (ADDI x_r, x0, r)
# x16     = -1     (ADDI x16, x0, 0xFFF → sign-extend → -1)
# x17     = LUI(0x7FFFF) + ADDI(-1) = 0x7FFFF000 - 1 = 0x7FFFEFFF
# x18     = LUI(0x80000) = 0x80000000
# x19     = 0      (not set in prologue)
# x20     = LUI(0x00010) = 0x10000
# x21     = LUI(0x12345) + ADDI(0x678) = 0x12345000 + 0x678 = 0x12345678
# x22-x31 = 22..31 (ADDI x_r, x0, r)

def _prologue_regs() -> list[int]:
    regs = [0] * 32
    for r in range(1, 16):
        regs[r] = r
    regs[16] = 0xFFFFFFFF          # -1
    regs[17] = 0x7FFFEFFF          # ~INT_MAX (LUI 0x7FFFF + ADDI -1)
    regs[18] = 0x80000000          # INT_MIN
    regs[20] = 0x00010000          # 0x10000
    regs[21] = 0x12345678
    for r in range(22, 32):
        regs[r] = r
    return regs


# Pre-computed start PC (program loaded at 0x100080, prologue has 31 instrs)
_PROLOGUE_LEN = 2 + 15 + 1 + 2 + 1 + 1 + 2 + 10  # = 34 instructions
_PROG_BASE    = 0x100080
_START_PC     = _PROG_BASE + _PROLOGUE_LEN * 4   # 0x100080 + 0x88 = 0x100108


class RISCVShadow:
    """Cycle-accurate-ish ISA simulator for coverage purposes."""

    def __init__(self):
        self.regs: list[int] = _prologue_regs()
        self.pc:   int       = _START_PC
        self._mem: dict[int, int] = {}  # addr → 32-bit word (word-granularity writes)

    def reset(self):
        self.regs = _prologue_regs()
        self.pc   = _START_PC
        self._mem.clear()

    # ── Memory helpers ────────────────────────────────────────────────────────

    def _mem_read_byte(self, addr: int) -> int:
        word_addr = addr & ~3
        shift     = (addr & 3) * 8
        word      = self._mem.get(word_addr, (word_addr ^ 0xDEADBEEF) & 0xFFFFFFFF)
        return (word >> shift) & 0xFF

    def _mem_read_half(self, addr: int) -> int:
        return self._mem_read_byte(addr) | (self._mem_read_byte(addr + 1) << 8)

    def _mem_read_word(self, addr: int) -> int:
        word_addr = addr & ~3
        if addr & 3:  # misaligned — read byte by byte
            b = [self._mem_read_byte(addr + i) for i in range(4)]
            return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
        return self._mem.get(word_addr, (word_addr ^ 0xDEADBEEF) & 0xFFFFFFFF)

    def _mem_write_byte(self, addr: int, val: int):
        word_addr = addr & ~3
        shift     = (addr & 3) * 8
        old = self._mem.get(word_addr, (word_addr ^ 0xDEADBEEF) & 0xFFFFFFFF)
        self._mem[word_addr] = (old & ~(0xFF << shift)) | ((val & 0xFF) << shift)

    def _mem_write_half(self, addr: int, val: int):
        self._mem_write_byte(addr,     val & 0xFF)
        self._mem_write_byte(addr + 1, (val >> 8) & 0xFF)

    def _mem_write_word(self, addr: int, val: int):
        word_addr = addr & ~3
        if addr & 3:
            for i in range(4):
                self._mem_write_byte(addr + i, (val >> (i * 8)) & 0xFF)
        else:
            self._mem[word_addr] = val & 0xFFFFFFFF

    # ── Register helpers ─────────────────────────────────────────────────────

    def _r(self, idx: int) -> int:
        return 0 if idx == 0 else self.regs[idx] & 0xFFFFFFFF

    def _w(self, idx: int, val: int):
        if idx != 0:
            self.regs[idx] = val & 0xFFFFFFFF

    # ── Sign extension ────────────────────────────────────────────────────────

    @staticmethod
    def _sext(val: int, bits: int) -> int:
        sign = 1 << (bits - 1)
        return (val & (sign - 1)) - (val & sign)

    @staticmethod
    def _s32(val: int) -> int:
        v = val & 0xFFFFFFFF
        return v if v < 0x80000000 else v - 0x100000000

    # ── Main decode+execute ──────────────────────────────────────────────────

    def step(self, insn: int) -> dict:
        """Execute one instruction word. Returns trace dict for CoverageModel."""
        insn &= 0xFFFFFFFF
        pc   = self.pc

        rd_addr   = 0; rs1_addr  = 0; rs2_addr  = 0
        rs1_rdata = 0; rs2_rdata = 0; rd_wdata  = 0
        mem_addr  = 0; trap      = False
        next_pc   = pc + 4

        if (insn & 0x3) != 0x3:
            # ── Compressed (16-bit) ───────────────────────────────────────
            next_pc = pc + 2
            hw = insn & 0xFFFF
            q  = hw & 0x3
            f3 = (hw >> 13) & 0x7

            if q == 0b01:
                if f3 == 0:   # C.ADDI
                    rd_addr = rs1_addr = (hw >> 7) & 0x1F
                    imm = self._sext((((hw >> 12) & 1) << 5) | ((hw >> 2) & 0x1F), 6)
                    rs1_rdata = self._r(rs1_addr)
                    rd_wdata  = (rs1_rdata + imm) & 0xFFFFFFFF
                    self._w(rd_addr, rd_wdata)
                elif f3 == 2:  # C.LI
                    rd_addr = (hw >> 7) & 0x1F
                    imm = self._sext((((hw >> 12) & 1) << 5) | ((hw >> 2) & 0x1F), 6)
                    rd_wdata = imm & 0xFFFFFFFF
                    self._w(rd_addr, rd_wdata)
                elif f3 in (1, 5):  # C.J (f3=5) / C.JAL (f3=1, RV32C only)
                    b   = hw
                    imm = (((b>>12)&1)<<11)|(((b>>11)&1)<<4)|(((b>>9)&3)<<8)| \
                          (((b>>8)&1)<<10)|(((b>>7)&1)<<6)|(((b>>6)&1)<<7)|   \
                          (((b>>3)&7)<<1)|(((b>>2)&1)<<5)
                    imm = self._sext(imm, 12)
                    if f3 == 1:  # C.JAL saves return addr to x1
                        rd_addr  = 1
                        rd_wdata = (pc + 2) & 0xFFFFFFFF
                        self._w(1, rd_wdata)
                    next_pc = (pc + imm) & 0xFFFFFFFF
                elif f3 == 6:  # C.BEQZ
                    rs1_addr  = 8 + ((hw >> 7) & 0x7)
                    rs1_rdata = self._r(rs1_addr)
                    imm = (((hw>>12)&1)<<8)|(((hw>>10)&3)<<3)|(((hw>>5)&3)<<6)| \
                          (((hw>>3)&3)<<1)|(((hw>>2)&1)<<5)
                    imm = self._sext(imm, 9)
                    if rs1_rdata == 0:
                        next_pc = (pc + imm) & 0xFFFFFFFF
                elif f3 == 7:  # C.BNEZ
                    rs1_addr  = 8 + ((hw >> 7) & 0x7)
                    rs1_rdata = self._r(rs1_addr)
                    imm = (((hw>>12)&1)<<8)|(((hw>>10)&3)<<3)|(((hw>>5)&3)<<6)| \
                          (((hw>>3)&3)<<1)|(((hw>>2)&1)<<5)
                    imm = self._sext(imm, 9)
                    if rs1_rdata != 0:
                        next_pc = (pc + imm) & 0xFFFFFFFF

            elif q == 0b00:
                rs1_addr  = 8 + ((hw >> 7) & 0x7)
                rs1_rdata = self._r(rs1_addr)
                if f3 == 2:   # C.LW
                    rd_addr  = 8 + ((hw >> 2) & 0x7)
                    offset   = (((hw>>10)&7)<<3) | (((hw>>6)&1)<<2) | (((hw>>5)&1)<<6)
                    mem_addr = (rs1_rdata + offset) & 0xFFFFFFFF
                    rd_wdata = self._mem_read_word(mem_addr)
                    self._w(rd_addr, rd_wdata)
                elif f3 == 6:  # C.SW
                    rs2_addr  = 8 + ((hw >> 2) & 0x7)
                    rs2_rdata = self._r(rs2_addr)
                    offset    = (((hw>>10)&7)<<3) | (((hw>>6)&1)<<2) | (((hw>>5)&1)<<6)
                    mem_addr  = (rs1_rdata + offset) & 0xFFFFFFFF
                    self._mem_write_word(mem_addr, rs2_rdata)

            elif q == 0b10:
                if f3 == 0:   # C.SLLI
                    rd_addr = rs1_addr = (hw >> 7) & 0x1F
                    shamt   = (((hw >> 12) & 1) << 5) | ((hw >> 2) & 0x1F)
                    rs1_rdata = self._r(rs1_addr)
                    rd_wdata  = (rs1_rdata << (shamt & 0x1F)) & 0xFFFFFFFF
                    self._w(rd_addr, rd_wdata)
                elif f3 == 4:
                    rs2_f = (hw >> 2) & 0x1F
                    b12   = (hw >> 12) & 1
                    rr    = (hw >> 7) & 0x1F
                    if b12 == 0 and rs2_f != 0:   # C.MV
                        rd_addr  = rr; rs2_addr = rs2_f
                        rs2_rdata = self._r(rs2_addr)
                        rd_wdata  = rs2_rdata
                        self._w(rd_addr, rd_wdata)
                    elif b12 == 1 and rs2_f != 0:  # C.ADD
                        rd_addr = rs1_addr = rr; rs2_addr = rs2_f
                        rs1_rdata = self._r(rs1_addr)
                        rs2_rdata = self._r(rs2_addr)
                        rd_wdata  = (rs1_rdata + rs2_rdata) & 0xFFFFFFFF
                        self._w(rd_addr, rd_wdata)

        else:
            # ── 32-bit instruction ────────────────────────────────────────
            op  = insn & 0x7F
            f3  = (insn >> 12) & 0x7
            f7  = (insn >> 25) & 0x7F
            rd_addr  = (insn >> 7)  & 0x1F
            rs1_addr = (insn >> 15) & 0x1F
            rs2_addr = (insn >> 20) & 0x1F
            rs1_rdata = self._r(rs1_addr)
            rs2_rdata = self._r(rs2_addr)

            if op == 0x33:   # ── R-type ───────────────────────────────────
                shamt = rs2_rdata & 0x1F
                s1 = self._s32(rs1_rdata); s2 = self._s32(rs2_rdata)
                if   f7 == 0x01:  # M-extension
                    if   f3 == 0: rd_wdata = (rs1_rdata * rs2_rdata) & 0xFFFFFFFF        # MUL
                    elif f3 == 1: rd_wdata = ((s1 * s2) >> 32) & 0xFFFFFFFF               # MULH
                    elif f3 == 2: rd_wdata = ((s1 * rs2_rdata) >> 32) & 0xFFFFFFFF        # MULHSU
                    elif f3 == 3: rd_wdata = ((rs1_rdata * rs2_rdata) >> 32) & 0xFFFFFFFF # MULHU
                    elif f3 == 4:                                                          # DIV
                        if rs2_rdata == 0: rd_wdata = 0xFFFFFFFF
                        elif rs1_rdata == 0x80000000 and rs2_rdata == 0xFFFFFFFF: rd_wdata = 0x80000000
                        else: rd_wdata = int(s1 / s2) & 0xFFFFFFFF
                    elif f3 == 5:                                                          # DIVU
                        rd_wdata = 0xFFFFFFFF if rs2_rdata == 0 else (rs1_rdata // rs2_rdata) & 0xFFFFFFFF
                    elif f3 == 6:                                                          # REM
                        if rs2_rdata == 0: rd_wdata = rs1_rdata
                        elif rs1_rdata == 0x80000000 and rs2_rdata == 0xFFFFFFFF: rd_wdata = 0
                        else: rd_wdata = int(s1 - int(s1/s2)*s2) & 0xFFFFFFFF
                    elif f3 == 7:                                                          # REMU
                        rd_wdata = rs1_rdata if rs2_rdata == 0 else (rs1_rdata % rs2_rdata) & 0xFFFFFFFF
                elif f7 == 0x00:
                    if   f3 == 0: rd_wdata = (rs1_rdata + rs2_rdata) & 0xFFFFFFFF  # ADD
                    elif f3 == 1: rd_wdata = (rs1_rdata << shamt) & 0xFFFFFFFF     # SLL
                    elif f3 == 2: rd_wdata = 1 if s1 < s2 else 0                   # SLT
                    elif f3 == 3: rd_wdata = 1 if rs1_rdata < rs2_rdata else 0     # SLTU
                    elif f3 == 4: rd_wdata = (rs1_rdata ^ rs2_rdata) & 0xFFFFFFFF  # XOR
                    elif f3 == 5: rd_wdata = (rs1_rdata >> shamt) & 0xFFFFFFFF     # SRL
                    elif f3 == 6: rd_wdata = (rs1_rdata | rs2_rdata) & 0xFFFFFFFF  # OR
                    elif f3 == 7: rd_wdata = (rs1_rdata & rs2_rdata) & 0xFFFFFFFF  # AND
                elif f7 == 0x20:
                    if   f3 == 0: rd_wdata = (rs1_rdata - rs2_rdata) & 0xFFFFFFFF  # SUB
                    elif f3 == 5:                                                    # SRA
                        rd_wdata = (s1 >> shamt) & 0xFFFFFFFF
                self._w(rd_addr, rd_wdata)

            elif op == 0x13:  # ── I-type arithmetic ────────────────────────
                imm   = self._sext((insn >> 20) & 0xFFF, 12)
                shamt = (insn >> 20) & 0x1F
                s1    = self._s32(rs1_rdata)
                if   f3 == 0: rd_wdata = (rs1_rdata + imm) & 0xFFFFFFFF      # ADDI
                elif f3 == 2: rd_wdata = 1 if s1 < imm else 0                 # SLTI
                elif f3 == 3: rd_wdata = 1 if rs1_rdata < (imm & 0xFFFFFFFF) else 0  # SLTIU
                elif f3 == 4: rd_wdata = (rs1_rdata ^ (imm & 0xFFFFFFFF)) & 0xFFFFFFFF  # XORI
                elif f3 == 6: rd_wdata = (rs1_rdata | (imm & 0xFFFFFFFF)) & 0xFFFFFFFF  # ORI
                elif f3 == 7: rd_wdata = (rs1_rdata & (imm & 0xFFFFFFFF)) & 0xFFFFFFFF  # ANDI
                elif f3 == 1: rd_wdata = (rs1_rdata << shamt) & 0xFFFFFFFF    # SLLI
                elif f3 == 5:
                    if (insn >> 25) & 0x7F == 0x00:
                        rd_wdata = (rs1_rdata >> shamt) & 0xFFFFFFFF           # SRLI
                    else:
                        rd_wdata = (self._s32(rs1_rdata) >> shamt) & 0xFFFFFFFF  # SRAI
                self._w(rd_addr, rd_wdata)

            elif op == 0x03:  # ── Loads ────────────────────────────────────
                imm      = self._sext((insn >> 20) & 0xFFF, 12)
                mem_addr = (rs1_rdata + imm) & 0xFFFFFFFF
                if   f3 == 0: rd_wdata = self._sext(self._mem_read_byte(mem_addr), 8) & 0xFFFFFFFF  # LB
                elif f3 == 1: rd_wdata = self._sext(self._mem_read_half(mem_addr), 16) & 0xFFFFFFFF # LH
                elif f3 == 2: rd_wdata = self._mem_read_word(mem_addr)                              # LW
                elif f3 == 4: rd_wdata = self._mem_read_byte(mem_addr)                              # LBU
                elif f3 == 5: rd_wdata = self._mem_read_half(mem_addr)                              # LHU
                self._w(rd_addr, rd_wdata)

            elif op == 0x23:  # ── Stores ───────────────────────────────────
                imm      = self._sext((((insn>>25)&0x7F)<<5) | ((insn>>7)&0x1F), 12)
                mem_addr = (rs1_rdata + imm) & 0xFFFFFFFF
                if   f3 == 0: self._mem_write_byte(mem_addr, rs2_rdata & 0xFF)
                elif f3 == 1: self._mem_write_half(mem_addr, rs2_rdata & 0xFFFF)
                elif f3 == 2: self._mem_write_word(mem_addr, rs2_rdata)
                rd_addr = 0; rd_wdata = 0

            elif op == 0x63:  # ── Branches ─────────────────────────────────
                imm = self._sext(
                    (((insn>>31)&1)<<12)|(((insn>>7)&1)<<11)|
                    (((insn>>25)&0x3F)<<5)|(((insn>>8)&0xF)<<1), 13)
                s1 = self._s32(rs1_rdata); s2 = self._s32(rs2_rdata)
                taken = False
                if   f3 == 0: taken = rs1_rdata == rs2_rdata          # BEQ
                elif f3 == 1: taken = rs1_rdata != rs2_rdata           # BNE
                elif f3 == 4: taken = s1 < s2                          # BLT
                elif f3 == 5: taken = s1 >= s2                         # BGE
                elif f3 == 6: taken = rs1_rdata < rs2_rdata            # BLTU
                elif f3 == 7: taken = rs1_rdata >= rs2_rdata           # BGEU
                next_pc = (pc + imm) & 0xFFFFFFFF if taken else pc + 4
                rd_addr = 0; rd_wdata = 0

            elif op == 0x6F:  # ── JAL ──────────────────────────────────────
                imm = self._sext(
                    (((insn>>31)&1)<<20)|(((insn>>12)&0xFF)<<12)|
                    (((insn>>20)&1)<<11)|(((insn>>21)&0x3FF)<<1), 21)
                rd_wdata = (pc + 4) & 0xFFFFFFFF
                next_pc  = (pc + imm) & 0xFFFFFFFF
                self._w(rd_addr, rd_wdata)

            elif op == 0x67:  # ── JALR ─────────────────────────────────────
                imm      = self._sext((insn >> 20) & 0xFFF, 12)
                rd_wdata = (pc + 4) & 0xFFFFFFFF
                next_pc  = (rs1_rdata + imm) & ~1 & 0xFFFFFFFF
                self._w(rd_addr, rd_wdata)

            elif op == 0x37:  # ── LUI ──────────────────────────────────────
                rd_wdata = insn & 0xFFFFF000
                self._w(rd_addr, rd_wdata)

            elif op == 0x17:  # ── AUIPC ─────────────────────────────────────
                rd_wdata = (pc + (insn & 0xFFFFF000)) & 0xFFFFFFFF
                self._w(rd_addr, rd_wdata)

            elif op == 0x73:  # ── System ───────────────────────────────────
                f12 = (insn >> 20) & 0xFFF
                if f3 == 0:
                    if   f12 == 0x000: trap = True   # ECALL
                    elif f12 == 0x001: trap = True   # EBREAK
                    # MRET, WFI: no-op in shadow
                elif f3 in (0x1, 0x2, 0x3, 0x5, 0x6, 0x7):  # CSR*
                    rd_wdata = 0   # simplified: CSR reads return 0
                    self._w(rd_addr, rd_wdata)
            # FENCE / FENCEI: no-op

        self.pc = next_pc & 0xFFFFFFFF

        return {
            "insn"      : insn,
            "rs1_addr"  : rs1_addr,
            "rs2_addr"  : rs2_addr,
            "rd_addr"   : rd_addr,
            "rs1_rdata" : rs1_rdata,
            "rs2_rdata" : rs2_rdata,
            "rd_wdata"  : rd_wdata,
            "pc_rdata"  : pc,
            "pc_wdata"  : next_pc,
            "mem_addr"  : mem_addr,
            "trap"      : trap,
        }

    # ── Program-level interface ──────────────────────────────────────────────

    def run_actions(self, actions: list) -> dict:
        """Sequential execution + trap handler + WFI (best shadow-RTL match).

        Sequential (not CF) because random JALR/JAL jumps outside program
        bounds cause CF to exit after ~76 steps, losing 400+ steps of coverage.
        RTL handles out-of-bounds jumps via trap → continues; shadow cannot
        simulate that faithfully, so sequential is a better approximation.
        Branch coverage bins (taken/ntaken) are still correct because step()
        evaluates branch conditions regardless of next-PC tracking.
        """
        machine_code = emit_program(actions)
        model = CoverageModel()
        for insn in machine_code:
            pc_before = self.pc
            trace = self.step(insn)
            model.sample(trace)
            if trace["trap"]:
                for ht in _trap_handler_traces(pc_before):
                    model.sample(ht)
                self.regs[10] = (pc_before + 4) & 0xFFFFFFFF
        # Always execute WFI at end (matches RTL: PROLOGUE + agent_machine + [WFI])
        trace = self.step(_WFI)
        model.sample(trace)
        covered, total = model.count_bins()
        return {
            "covered": covered,
            "total"  : total,
            "pct"    : 100.0 * covered / max(total, 1),
            "hits"   : list(model.hit_set()),
        }

    def run_machine_code_cf(self, machine_code: list) -> dict:
        """Execute following actual branch/jump control flow.

        Each action occupies exactly one 4-byte slot in the program image.
        PC delta after each instruction determines the next slot index:
          delta == 4  → sequential (next slot)
          delta == 2  → compressed instruction (still advance one slot,
                        upper half 0x0000 is skipped as in real HW)
          delta == 8  → branch/JAL taken (skip next slot)
          delta == N  → jump to slot index = (current_PC + delta - _START_PC) // 4
        Execution stops when the slot index goes outside [0, len(machine_code)).
        Appends WFI at end to match RTL program layout (test_run_for_l9.py).
        Simulates trap handler on ECALL/EBREAK to match RTL behavior.
        """
        machine_code = list(machine_code) + [_WFI]
        n = len(machine_code)
        model = CoverageModel()
        max_steps = n * 3   # bound against infinite loops

        idx = 0
        steps = 0
        while 0 <= idx < n and steps < max_steps:
            insn = machine_code[idx]
            pc_before = self.pc
            trace = self.step(insn)
            model.sample(trace)
            steps += 1

            # Simulate trap handler on ECALL/EBREAK (matches RTL mtvec=0x00200000)
            if trace["trap"]:
                for ht in _trap_handler_traces(pc_before):
                    model.sample(ht)
                self.regs[10] = (pc_before + 4) & 0xFFFFFFFF

            pc_after = self.pc
            raw_delta = (pc_after - pc_before) & 0xFFFFFFFF
            if raw_delta >= 0x80000000:          # negative (backward jump)
                raw_delta -= 0x100000000

            if raw_delta == 2:                   # 16-bit compressed: one slot
                idx += 1
            elif raw_delta == 0:                 # should not happen; guard
                idx += 1
            else:
                idx += raw_delta // 4

        covered, total = model.count_bins()
        return {
            "covered": covered,
            "total"  : total,
            "pct"    : 100.0 * covered / max(total, 1),
            "hits"   : list(model.hit_set()),
        }

    def run_machine_code(self, machine_code: list) -> dict:
        """Sequential execution (legacy, no control flow). Use run_machine_code_cf."""
        model = CoverageModel()
        for insn in machine_code:
            trace = self.step(insn)
            model.sample(trace)
        covered, total = model.count_bins()
        return {
            "covered": covered,
            "total"  : total,
            "pct"    : 100.0 * covered / max(total, 1),
            "hits"   : list(model.hit_set()),
        }

    def run_actions_traced(self, actions: list) -> tuple[dict, CoverageModel]:
        """Like run_actions but also returns the CoverageModel for inspection."""
        self.reset()
        model = CoverageModel()
        machine_code = emit_program(actions)
        for insn in machine_code:
            trace = self.step(insn)
            model.sample(trace)
        covered, total = model.count_bins()
        result = {
            "covered": covered,
            "total"  : total,
            "pct"    : 100.0 * covered / max(total, 1),
            "hits"   : list(model.hit_set()),
        }
        return result, model


# ── Verification helper (compare shadow vs Verilator) ────────────────────────

def verify_vs_verilator(actions: list, driver: str = "test_run_for_l9") -> dict:
    """Run both shadow and Verilator on the same actions; compare coverage."""
    from env_l9 import run_program as vtop_run

    shadow  = RISCVShadow()
    shadow.reset()
    shadow_result = shadow.run_actions(actions)

    vtop_result = vtop_run(actions, driver=driver)
    if vtop_result is None:
        return {"error": "vtop failed"}

    s_hits = set(shadow_result["hits"])
    v_hits = set(vtop_result["hits"])
    only_shadow = s_hits - v_hits
    only_vtop   = v_hits - s_hits

    return {
        "shadow_pct"  : shadow_result["pct"],
        "vtop_pct"    : vtop_result["pct"],
        "jaccard"     : len(s_hits & v_hits) / max(len(s_hits | v_hits), 1),
        "only_shadow" : len(only_shadow),
        "only_vtop"   : len(only_vtop),
        "examples_only_vtop" : list(only_vtop)[:10],
    }


if __name__ == "__main__":
    import random
    shadow = RISCVShadow()
    actions = [(random.randint(0, 69), random.randint(0, 31),
                random.randint(0, 31), random.randint(0, 31),
                random.randint(0, 4)) for _ in range(256)]
    result = shadow.run_actions(actions)
    print(f"Shadow self-test: {result['covered']}/{result['total']} = {result['pct']:.2f}%")
