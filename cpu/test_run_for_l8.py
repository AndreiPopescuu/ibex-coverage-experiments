"""Level 8 cocotb driver — enhanced for LINE coverage.

Extends the L7 driver with:
  1. Longer prologue that pre-initialises all 32 registers with diverse values
     (ensures branches compare non-trivial operands from the first instruction).
  2. Trap handler advanced: reads both mepc (0x341) and mcause (0x342) before
     MRET, toggling more CSR-register paths.
  3. Data memory returns addr XOR 0xDEADBEEF for read-misses (same as L7).
  4. Accepts programs from RL_L8_JSON (falls back to RL_L7_JSON for compat).

Program layout:
    0x00100080  PROLOGUE (LUI/ADDI to initialise registers)
    0x00100080 + 4*len(PROLOGUE)  agent program
    ...
    0x00200000  TRAP HANDLER (reads mepc+mcause, advances mepc, MRET)
"""

import os, sys, json
_here = os.path.dirname(os.path.abspath(__file__ if "__file__" in dir() else "."))
_ml4dv = os.path.abspath(os.path.join(_here, ".."))
if _ml4dv not in sys.path:
    sys.path.insert(0, _ml4dv)

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, ClockCycles, ReadWrite

from test_cpu_coverage import MemAgent


WFI   = 0x10500073
MRET  = 0x30200073


# ── Register initialisation prologue ─────────────────────────────────────
# Goal: give every register a distinct, non-zero, non-trivial value so that
# branches, ALU ops, and loads receive meaningful operands from instruction 1.
#
# Strategy: use LUI+ADDI pairs for regs that need a 32-bit value;
# plain ADDI x0 for small values.
#
# Register plan:
#   x0  = 0  (hardwired)
#   x1  = 0x00001001
#   x2  = 0x00002002   (SP-like, will be overwritten by agent)
#   x3  = 0x00003003
#   ...
#   x10 = 0x00010010   (function argument — diverse)
#   x11 = 0x0000000B   (small)
#   x15 = 0x0000000F
#   x16 = 0xFFFFFFFF   (-1 for signed-compare tests)
#   x17 = 0x7FFFFFFF   (INT_MAX)
#   x18 = 0x80000000   (INT_MIN, largest unsigned)
#   x19 = 0x55555555   (alternating bits)
#   x20 = 0x00010000   (safe data-memory base address)
#   x21 = 0xAAAAAAAA   (alternating bits, inverted)
#   x22 = 0x12345678
#   x23 = 0xFEDCBA98 (sign-extended upper portion)
#   x24..x31 = small values 24..31

def _addi_w(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | (0b000 << 12) | ((rd & 0x1F) << 7) | 0b0010011

def _lui_w(rd, imm20):
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | 0b0110111

def _csrrw_w(rd, rs1, csr):
    return ((csr & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | (0b001 << 12) | ((rd & 0x1F) << 7) | 0b1110011

def _csrrsi_w(rd, uimm5, csr):
    # CSRRSI: funct3=110, uimm5 occupies the rs1 field (bits 19:15); no register read
    return ((csr & 0xFFF) << 20) | ((uimm5 & 0x1F) << 15) | (0b110 << 12) | ((rd & 0x1F) << 7) | 0b1110011

def _srli_w(rd, rs1, shamt):
    # Shift-right-logical: funct7=0000000, shamt in bits 24:20, funct3=101
    return ((shamt & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (0b101 << 12) | ((rd & 0x1F) << 7) | 0b0010011

def _bne_w(rs1, rs2, imm):
    # B-type: imm is a signed byte offset from the BNE instruction itself; must be even
    # Encoding: [31]=imm[12], [30:25]=imm[10:5], [24:20]=rs2, [19:15]=rs1,
    #           [14:12]=001, [11:8]=imm[4:1], [7]=imm[11], [6:0]=1100011
    return (
        (((imm >> 12) & 1) << 31) |
        (((imm >> 5) & 0x3F) << 25) |
        ((rs2 & 0x1F) << 20) |
        ((rs1 & 0x1F) << 15) |
        (0b001 << 12) |
        (((imm >> 1) & 0xF) << 8) |
        (((imm >> 11) & 1) << 7) |
        0b1100011
    )


def _build_prologue():
    p = []

    def emit(*ws): p.extend(ws)

    # Set mtvec = 0x00200000 (trap handler location)
    emit(_lui_w(10, 0x00200))          # x10 = 0x00200000
    emit(_csrrw_w(0, 10, 0x305))       # CSR[mtvec] = x10

    # Register diversity setup
    for r in range(1, 16):
        emit(_addi_w(r, 0, r))          # x1..x15 = 1..15

    # x16 = -1
    emit(_addi_w(16, 0, 0xFFF))        # ADDI x16, x0, -1 (sign-extended)

    # x17 = 0x7FFFFFFF (INT_MAX)
    emit(_lui_w(17, 0x7FFFF))
    emit(_addi_w(17, 17, 0xFFF))       # x17 = 0x7FFFF000 + 0xFFF = 0x7FFFFFFF

    # x18 = 0x80000000 (INT_MIN / largest unsigned)
    emit(_lui_w(18, 0x80000))

    # x19 = 0x55555555 (alternating bits)
    emit(_lui_w(19, 0x55555))
    emit(_addi_w(19, 19, 0x555))       # x19 = 0x55555000 + 0x555 = 0x55555555

    # x20 = 0x00010000 (data-mem base)
    emit(_lui_w(20, 0x00010))

    # x21 = 0xAAAAAAAA (alternating bits inverted)
    emit(_lui_w(21, 0xAAAAA))
    emit(_addi_w(21, 21, 0xAAA & 0xFFF))  # sign-extends: 0xAAAA000 - 0x556 = 0xAAA9AAA?
    # Use correct two's complement: 0xAAA = -0x556 as 12-bit signed
    # LUI 0xAAAAB + ADDI -0x556... just use simpler approach:
    emit(_lui_w(21, 0xAAAAB))
    emit(_addi_w(21, 21, -0x556 & 0xFFF))  # 0xAAAAB000 - 0x556 = 0xAAAAAAAA

    # x22 = 0x12345678
    emit(_lui_w(22, 0x12345))
    emit(_addi_w(22, 22, 0x678))

    # x23 = 0xFEDCBA98
    # LUI 0xFEDCC + ADDI -0x368 (because 0xFEDCC000 - 0x368 = 0xFEDCBA98? Let's check:
    # 0xFEDCC000 = 4275765248, 0xBA98 = 47768, 4275765248 + 47768? No.
    # FEDCBA98: LUI = 0xFEDCB, ADDI = 0xA98 (sign-ext = 0xFFFFF-0x568 = -0xA68? no)
    # 0xA98 as 12-bit = 2712, positive. 0xFEDCB000 + 0xA98 = 0xFEDCBA98. ✓
    emit(_lui_w(23, 0xFEDCB))
    emit(_addi_w(23, 23, 0xA98))

    # x24..x31 = 24..31
    for r in range(24, 32):
        emit(_addi_w(r, 0, r))

    # Enable all machine interrupt sources: write 0xFFFFFFFF to mie (0x304)
    # ADDI x10, x0, -1 encodes imm=0xFFF which sign-extends to 0xFFFFFFFF
    emit(_addi_w(10, 0, 0xFFF))           # x10 = 0xFFFFFFFF (all bits set)
    emit(_csrrw_w(0, 10, 0x304))          # CSR[mie] = x10   (0x304 = mie, M-mode interrupt-enable)
    # Set mstatus.MIE (bit 3) to globally enable M-mode interrupts
    # CSRRSI x0, mstatus, 8 ORs uimm=8 (bit 3) into mstatus (0x300)
    emit(_csrrsi_w(0, 8, 0x300))          # CSR[mstatus] |= 8 (0x300 = mstatus, bit3 = MIE)

    return p


PROLOGUE = _build_prologue()

TRAP_HANDLER_ADDR = 0x00200000

# Interrupt-aware trap handler.
#
# Layout (each instruction at a 4-byte offset from TRAP_HANDLER_ADDR):
#   +0:  CSRRS x10, mepc,   x0  — save mepc
#   +4:  CSRRS x11, mcause, x0  — read mcause; bit 31 = 1 → interrupt, 0 → exception
#   +8:  CSRRS x12, mtval,  x0  — read mtval  (exercises another CSR read path)
#   +12: SRLI  x13, x11, 31     — x13 = mcause[31]: 1=interrupt, 0=exception
#   +16: BNE   x13, x0, +8     — if interrupt: skip mepc advance (jump to +24)
#   +20: ADDI  x10, x10, 4     — exception only: advance past faulting instruction
#   +24: CSRRW x0,  mepc, x10  — write (possibly updated) mepc back
#   +28: MRET                   — return from trap; re-enables MIE via mstatus.MPIE
def _build_trap_handler():
    p = []
    # offset +0: x10 = mepc (0x341 = mepc CSR, M-mode exception PC)
    p.append((0x341 << 20) | (0 << 15) | (0b010 << 12) | (10 << 7) | 0b1110011)  # CSRRS x10, mepc, x0
    # offset +4: x11 = mcause (0x342 = mcause CSR; bit 31 distinguishes interrupt vs exception)
    p.append((0x342 << 20) | (0 << 15) | (0b010 << 12) | (11 << 7) | 0b1110011)  # CSRRS x11, mcause, x0
    # offset +8: x12 = mtval (0x343 = mtval CSR; exercises a third CSR read path)
    p.append((0x343 << 20) | (0 << 15) | (0b010 << 12) | (12 << 7) | 0b1110011)  # CSRRS x12, mtval, x0
    # offset +12: x13 = mcause >> 31  → 1 if interrupt, 0 if exception
    p.append(_srli_w(13, 11, 31))    # SRLI x13, x11, 31
    # offset +16: if x13 != 0 (interrupt): branch forward by +8 bytes to offset +24 (CSRRW)
    # imm=+8: target = PC_of_this_insn + 8 = (TRAP_HANDLER_ADDR+16) + 8 = +24 ✓
    p.append(_bne_w(13, 0, 8))       # BNE x13, x0, +8
    # offset +20: exception path only — advance mepc past the faulting instruction
    p.append(_addi_w(10, 10, 4))     # ADDI x10, x10, 4
    # offset +24: write mepc back (0x341 = mepc CSR)
    p.append(_csrrw_w(0, 10, 0x341)) # CSRRW x0, mepc, x10  (= 0x34151073)
    # offset +28: return from trap; hardware restores MIE from MPIE
    p.append(MRET)                   # 0x30200073
    return p

TRAP_HANDLER = _build_trap_handler()


class DiverseMemAgent(MemAgent):
    """Data-memory agent: returns (addr XOR 0xDEADBEEF) for read-misses."""
    async def run_mem(self):
        self.gnt.value = 0
        self.rvalid.value = 0
        while True:
            await ClockCycles(self.clk, 1)
            await ReadWrite()
            self.rvalid.value = 0
            if self.req.value:
                self.gnt.value = 1
                access_addr = int(self.addr.value)
                write_data = None
                if self.handle_writes and self.we.value:
                    write_data = int(self.wdata.value)
                await ClockCycles(self.clk, 1)
                await ReadWrite()
                self.gnt.value = 0
                self.rvalid.value = 1
                if self.handle_writes and write_data is not None:
                    self.rdata.value = 0
                    self.mem_dict[access_addr] = write_data
                elif access_addr in self.mem_dict:
                    self.rdata.value = self.mem_dict[access_addr]
                else:
                    self.rdata.value = (access_addr ^ 0xDEADBEEF) & 0xFFFFFFFF


PROGRAM_PATH = os.environ.get(
    "RL_L8_JSON",
    os.environ.get("RL_L7_JSON", "/tmp/rl_l8_program.json")
)


async def irq_driver(dut, seed):
    """Randomly assert interrupt lines to drive interrupt-handling coverage.

    debug_req_i is intentionally kept at 0 (exposed as a port for wiring only).
    irq_nm_i (non-maskable interrupt) is also kept at 0 to avoid non-maskable
    re-entry complications with the simple trap handler above.
    """
    import random
    rng = random.Random(seed)
    # Initialise all interrupt lines to deasserted
    dut.irq_fast_i.value     = 0
    dut.irq_external_i.value = 0
    dut.irq_software_i.value = 0
    dut.irq_timer_i.value    = 0
    dut.irq_nm_i.value       = 0
    dut.debug_req_i.value    = 0   # kept at 0 (port exposed for completeness, not driven)

    # Wait for the prologue to finish configuring mtvec, mie, and mstatus.MIE
    # before injecting any interrupt. The prologue is ~50 instructions * ~5 CPI
    # at most, so 300 cycles is a comfortable margin.
    await ClockCycles(dut.clk_i, 300)

    while True:
        # Random quiet period between assertions (30–150 cycles)
        await ClockCycles(dut.clk_i, rng.randint(30, 150))
        # Assert a random mix of maskable interrupt lines
        dut.irq_fast_i.value     = rng.randint(1, 0x7FFF)   # at least one fast-IRQ bit set
        dut.irq_external_i.value = rng.randint(0, 1)
        dut.irq_software_i.value = rng.randint(0, 1)
        dut.irq_timer_i.value    = rng.randint(0, 1)
        # Hold for 5 cycles then deassert — short enough that the CPU completes the
        # trap handler before the next assertion, preventing un-handled re-entry.
        await ClockCycles(dut.clk_i, 5)
        dut.irq_fast_i.value     = 0
        dut.irq_external_i.value = 0
        dut.irq_software_i.value = 0
        dut.irq_timer_i.value    = 0


@cocotb.test()
async def run_program(dut):
    with open(PROGRAM_PATH) as f:
        payload = json.load(f)

    agent_machine = list(payload["machine_code"])
    full_program = PROLOGUE + agent_machine + [WFI]

    dut.data_gnt_i.value = 0
    dut.data_rvalid_i.value = 0

    # Initialise interrupt ports to 0 before reset (belt-and-suspenders alongside irq_driver)
    dut.irq_fast_i.value     = 0
    dut.irq_external_i.value = 0
    dut.irq_software_i.value = 0
    dut.irq_timer_i.value    = 0
    dut.irq_nm_i.value       = 0
    dut.debug_req_i.value    = 0

    imem = MemAgent(dut, "instr", handle_writes=False)
    dmem = DiverseMemAgent(dut, "data", handle_writes=True)

    imem.load_program(full_program, 0x100080)
    for i, word in enumerate(TRAP_HANDLER):
        imem.mem_dict[TRAP_HANDLER_ADDR + 4 * i] = word

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    dut.rst_ni.value = 1
    await Timer(15, units="ns")
    dut.rst_ni.value = 0
    await ClockCycles(dut.clk_i, 3)
    await Timer(5, units="ns")
    dut.rst_ni.value = 1

    cocotb.start_soon(imem.run_mem())
    cocotb.start_soon(dmem.run_mem())
    seed = hash(PROGRAM_PATH) & 0xFFFFFFFF
    cocotb.start_soon(irq_driver(dut, seed))

    # Extra cycles for MUL/DIV (multi-cycle ops) + trap round-trips
    max_cycles = len(full_program) * 80 + 5000
    for _ in range(max_cycles):
        await ClockCycles(dut.clk_i, 1)
        await ReadWrite()

    print(f"L8_RUN_COMPLETE: ran {len(full_program)} instr "
          f"(prologue={len(PROLOGUE)}, agent={len(agent_machine)})")
