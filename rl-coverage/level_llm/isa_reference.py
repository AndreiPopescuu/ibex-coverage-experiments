"""isa_reference.py — turns the L9 codec's encoding rules into an LLM-facing
cheat sheet, and validates/converts LLM-emitted action dicts into the
(op, rd, rs1, rs2, imm_bucket) tuples that starter.run_and_check expects.

The action space is NOT free-form assembly: every instruction is a 5-tuple
where `imm_bucket` is an index (0-4) into a small fixed table whose contents
depend on the instruction class. This module documents those tables (reading
them straight from codec_l9 / codec_l8 / codec_l7 — the ground truth used by
run_program) so an LLM can target a specific bin without guessing encodings.
"""

import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
for d in ("level9_ops", "level8_ops", "level7_stimulus"):
    sys.path.insert(0, str(THIS.parent / d))

import codec_l9
from codec_l9 import Op, N_OPS, IMM_BUCKETS, L9_CSRS  # noqa: E402

# ── Op name table (mirrors starter.py constants, extended with L9 compressed ops) ──
OP_NAMES = {op.name: int(op) for op in Op}

# ── Immediate tables, transcribed from the codecs but verified below ──────────
IMM_BUCKET_VALUES   = [-2048, -100, 0, 100, 2047]          # I-type ALU / loads / stores
SHAMT_BUCKET_VALUES = [0, 8, 16, 24, 31]                    # SLLI / SRLI / SRAI
BRANCH_BUCKET_OFFSETS = [4, 8, 12, 16, 20]                  # BEQ..BGEU (forward, bytes)
JAL_BUCKET_OFFSETS    = [4, 8, 12, 16, 24]                  # JAL (forward, bytes)
JALR_OFFSET_BUCKETS   = [0, 4, 8, -4, 0]                    # JALR (bytes)
LUI_IMM_BUCKETS       = [0x00001, 0x12345, 0xABCDE, 0xFFFFF, 0x80000]
AUIPC_IMM_BUCKETS     = [0x00000, 0x00001, 0x12345, 0xABCDE, 0xFFFFF]

R_TYPE = {"ADD","SUB","SLL","SLT","SLTU","XOR","SRL","SRA","OR","AND",
          "MUL","MULH","MULHSU","MULHU","DIV","DIVU","REM","REMU"}
I_TYPE_IMM = {"ADDI","SLTI","SLTIU","XORI","ORI","ANDI","LB","LH","LW","LBU","LHU",
              "SB","SH","SW"}
SHIFTS  = {"SLLI","SRLI","SRAI"}
BRANCHES = {"BEQ","BNE","BLT","BGE","BLTU","BGEU"}
CSR_REG_OPS = {"CSRRW": 0b001, "CSRRS": 0b010, "CSRRC": 0b011}
CSR_IMM_OPS = {"CSRRWI": 0b101, "CSRRSI": 0b110, "CSRRCI": 0b111}

# ── Registers pre-loaded by the test harness prologue (test_run_for_l8.py) ────
# These are available as operands from instruction 0 onward — no need to
# construct them with LUI/ADDI first.
PRELOADED_REGS = {
    1: "0x00001001",  2: "0x00002002 (overwritten by SP use)", 3: "0x00003003",
    10: "0x00010010", 11: "0x0000000B", 15: "0x0000000F",
    16: "0xFFFFFFFF (-1, all bits set)",
    17: "0x7FFFFFFF (INT_MAX)",
    18: "0x80000000 (INT_MIN / sign bit only)",
    19: "0x55555555 (alternating bits 0101...)",
    20: "0x00010000 (safe data-memory base address)",
    21: "0xAAAAAAAA (alternating bits 1010..., inverted)",
    22: "0x12345678",
    23: "0xFEDCBA98",
}
for _i in range(24, 32):
    PRELOADED_REGS[_i] = f"0x{_i:08x} (small value = register index)"


# ── CSR routing table — brute-forced against the REAL encoder (ground truth) ──
def _decode_csr_addr(word: int) -> int:
    return (word >> 20) & 0xFFF


def build_csr_route_table():
    """For every CSR in L9_CSRS, find (op, rs1, imm_bucket) combos that route
    a CSRRW*/CSRRS*/CSRRC* instruction to it, by brute-forcing the encoder."""
    routes = {csr: [] for csr in L9_CSRS}
    for op_name, op_num in list(CSR_REG_OPS.items()) + list(CSR_IMM_OPS.items()):
        op_i = OP_NAMES[op_name]
        for rs1 in range(32):
            for ib in range(IMM_BUCKETS):
                word = codec_l9.encode(op_i, 5, rs1, 6, ib)
                csr = _decode_csr_addr(word)
                if csr in routes and len(routes[csr]) < 3:
                    routes[csr].append((op_name, rs1, ib))
    return routes


CSR_NAMES = {
    0x340: "mscratch", 0x341: "mepc", 0x342: "mcause", 0x343: "mtval",
    0xF11: "mvendorid", 0xF12: "marchid", 0xF13: "mimpid", 0xF14: "mhartid",
    0xF15: "mconfigptr", 0xB00: "mcycle", 0xB02: "minstret", 0xB80: "mcycleh",
    0xB82: "minstreth", 0x320: "mcountinhibit",
    0xB03: "mhpmcounter3", 0xB04: "mhpmcounter4", 0xB05: "mhpmcounter5",
    0xB06: "mhpmcounter6", 0xB07: "mhpmcounter7", 0xB08: "mhpmcounter8",
    0x323: "mhpmevent3", 0x324: "mhpmevent4", 0x325: "mhpmevent5",
    0x326: "mhpmevent6", 0x327: "mhpmevent7", 0x328: "mhpmevent8",
    0xC00: "cycle", 0xC01: "time", 0xC02: "instret",
    0x300: "mstatus", 0x304: "mie", 0x305: "mtvec",
}


def cheat_sheet_text(target_csrs=None) -> str:
    """Render the full encoding reference as prompt-ready text.

    target_csrs: optional list of CSR addresses to show routes for (keeps the
    prompt short — showing all 32 would be wasteful for groups that don't
    touch CSRs).
    """
    lines = []
    lines.append("ACTION FORMAT")
    lines.append("Each instruction you emit is a 5-tuple: [op, rd, rs1, rs2, imm_bucket]")
    lines.append("  op          : integer 0-82, selects the instruction (see OP TABLE)")
    lines.append("  rd, rs1, rs2: integer register numbers 0-31 (x0 is hardwired to zero)")
    lines.append("  imm_bucket  : integer 0-4 — an INDEX into a small fixed table of")
    lines.append("                immediate values. You CANNOT pick an arbitrary immediate;")
    lines.append("                you can only pick which of 5 pre-baked values to use,")
    lines.append("                and the table differs per instruction class (below).")
    lines.append("")
    lines.append("OP TABLE (name -> number); use the NAME in your JSON, we translate it")
    name_lines = []
    for name, num in sorted(OP_NAMES.items(), key=lambda kv: kv[1]):
        name_lines.append(f"{name}={num}")
    for i in range(0, len(name_lines), 8):
        lines.append("  " + "  ".join(name_lines[i:i+8]))
    lines.append("")
    lines.append("IMMEDIATE TABLES — what imm_bucket 0,1,2,3,4 actually produce:")
    lines.append(f"  R-type ALU/MUL/DIV ({', '.join(sorted(R_TYPE))}): imm_bucket is IGNORED")
    lines.append(f"  I-type ALU/loads/stores ({', '.join(sorted(I_TYPE_IMM))}):")
    lines.append(f"    imm_bucket -> immediate value: {dict(enumerate(IMM_BUCKET_VALUES))}")
    lines.append(f"  Shifts (SLLI, SRLI, SRAI):")
    lines.append(f"    imm_bucket -> shift amount:    {dict(enumerate(SHAMT_BUCKET_VALUES))}")
    lines.append(f"  Branches (BEQ, BNE, BLT, BGE, BLTU, BGEU):")
    lines.append(f"    imm_bucket -> forward byte offset: {dict(enumerate(BRANCH_BUCKET_OFFSETS))}")
    lines.append(f"  JAL:")
    lines.append(f"    imm_bucket -> forward byte offset: {dict(enumerate(JAL_BUCKET_OFFSETS))}")
    lines.append(f"  JALR:")
    lines.append(f"    imm_bucket -> byte offset:         {dict(enumerate(JALR_OFFSET_BUCKETS))}")
    lines.append(f"  LUI rd, imm_bucket  ->  rd = (table[imm_bucket] << 12):")
    lines.append(f"    imm_bucket -> 20-bit upper imm: " +
                 ", ".join(f"{i}:0x{v:05x}->rd=0x{(v<<12)&0xFFFFFFFF:08x}"
                           for i, v in enumerate(LUI_IMM_BUCKETS)))
    lines.append(f"  AUIPC rd, imm_bucket  ->  rd = PC + (table[imm_bucket] << 12):")
    lines.append(f"    imm_bucket -> 20-bit upper imm: " +
                 ", ".join(f"{i}:0x{v:05x}" for i, v in enumerate(AUIPC_IMM_BUCKETS)))
    lines.append("")
    lines.append("CSR INSTRUCTIONS — the target CSR is NOT picked directly; it is derived:")
    lines.append("  CSRRW/CSRRS/CSRRC  (op 27/28/29):")
    lines.append("    target_csr_index = (imm_bucket*7 + rs1) % 32   (picks from a 32-entry table)")
    lines.append("    rs1 = register whose value is written/set/cleared into the CSR")
    lines.append("    rd  = receives the CSR's OLD value")
    lines.append("  *** CRITICAL: rs1 serves DUAL ROLE — it BOTH routes to the target CSR AND")
    lines.append("      is the source register whose value gets written. You CANNOT choose rs1")
    lines.append("      freely for its value; you MUST use the rs1 from the verified routes table")
    lines.append("      below. To write a specific value: first load that value into the register")
    lines.append("      named in the route (e.g. route says rs1=6 → do ADDI x6,x0,100 first,")
    lines.append("      then CSRRW with rs1=6). Choosing a different rs1 silently redirects the")
    lines.append("      instruction to a DIFFERENT CSR, not mcountinhibit. ***")
    lines.append("  CSRRWI/CSRRSI/CSRRCI  (op 66/67/68):")
    lines.append("    target_csr_index = (imm_bucket*7 + (rs1 % 5)) % 32")
    lines.append("    write_value (5-bit immediate) = (imm_bucket*3 + (rs1 % 5)) & 0x1F")
    lines.append("    rd  = receives the CSR's OLD value (rs1 only affects routing, not data)")
    lines.append("")
    lines.append("  VERIFIED ROUTES to specific CSRs — [op_name, rs1, imm_bucket] (pick one):")
    routes = build_csr_route_table()
    csr_list = target_csrs if target_csrs else sorted(L9_CSRS)
    for csr in csr_list:
        name = CSR_NAMES.get(csr, f"0x{csr:03x}")
        rs = routes.get(csr, [])
        if rs:
            rs_txt = ", ".join(f"{op}(rs1={r1},imm_bucket={ib})" for op, r1, ib in rs)
            lines.append(f"    {name:14s} (0x{csr:03x}): {rs_txt}")
    lines.append("")
    lines.append("PRE-LOADED REGISTERS — the test harness initialises these BEFORE your")
    lines.append("program runs (test_run_for_l8.py prologue). Use them directly as operands")
    lines.append("instead of constructing constants with LUI:")
    for reg, desc in PRELOADED_REGS.items():
        lines.append(f"    x{reg:<3d} = {desc}")
    lines.append("")
    lines.append("EXCEPTION HANDLING IS ALREADY SET UP — DO NOT install your own mtvec/handler.")
    lines.append("  The harness pre-installs mtvec=0x00200000 and a trap handler there that")
    lines.append("  reads mepc/mcause/mtval, advances mepc by 4, and executes MRET. You can")
    lines.append("  safely emit ECALL, EBREAK, illegal instructions, or misaligned loads/")
    lines.append("  stores — they will trap, the handler returns, and your program continues")
    lines.append("  with the NEXT instruction you emitted (not the one that trapped).")
    lines.append("")
    lines.append("MEMORY MODEL: unwritten data addresses return (address XOR 0xDEADBEEF) on")
    lines.append("read. Programs run starting at 0x80000000 in M-mode. x20 = 0x00010000 is a")
    lines.append("pre-loaded safe base address for loads/stores.")
    return "\n".join(lines)


# ── Validation / conversion: LLM JSON action -> (op, rd, rs1, rs2, imm_bucket) ──
def parse_action(d: dict) -> tuple:
    """Convert one LLM-emitted action dict into a validated 5-tuple.
    Raises ValueError with a message suitable for feeding back to the LLM."""
    op_field = d.get("op")
    if isinstance(op_field, str):
        name = op_field.strip().upper()
        if name not in OP_NAMES:
            raise ValueError(f"unknown op name {op_field!r} (valid names: {sorted(OP_NAMES)})")
        op = OP_NAMES[name]
    elif isinstance(op_field, int):
        op = op_field
    else:
        raise ValueError(f"action missing/invalid 'op' field: {d!r}")

    if not (0 <= op < N_OPS):
        raise ValueError(f"op {op} out of range 0-{N_OPS-1}")

    out = []
    for key in ("rd", "rs1", "rs2", "imm_bucket"):
        v = d.get(key, 0)
        if not isinstance(v, int):
            raise ValueError(f"field {key!r} must be an integer, got {v!r}")
        out.append(v)
    rd, rs1, rs2, ib = out
    if not (0 <= rd < 32 and 0 <= rs1 < 32 and 0 <= rs2 < 32):
        raise ValueError(f"register fields must be 0-31: rd={rd} rs1={rs1} rs2={rs2}")
    if not (0 <= ib < IMM_BUCKETS):
        raise ValueError(f"imm_bucket must be 0-{IMM_BUCKETS-1}, got {ib}")
    return (op, rd, rs1, rs2, ib)


def parse_program(actions_json) -> list[tuple]:
    """Convert a list of LLM-emitted action dicts into validated 5-tuples.
    Raises ValueError on the first invalid entry (index included in message)."""
    if not isinstance(actions_json, list) or not actions_json:
        raise ValueError("expected a non-empty JSON list of action objects")
    out = []
    for i, d in enumerate(actions_json):
        if not isinstance(d, dict):
            raise ValueError(f"action[{i}] must be an object, got {d!r}")
        try:
            out.append(parse_action(d))
        except ValueError as e:
            raise ValueError(f"action[{i}] ({d!r}): {e}")
    return out


if __name__ == "__main__":
    print(cheat_sheet_text())
