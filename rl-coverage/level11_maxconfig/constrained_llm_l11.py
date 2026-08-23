"""constrained_llm_l11.py — CRT constraints derived by reading Ibex RTL directly.
Zero riscv-dv / testlist.yaml / UVM knowledge was used anywhere in this
pipeline — every claim below is traceable to a specific RTL file:line, cross
checked against codec_l11.py's op indices and L11_CSRS bucket table.

HISTORY
  Pass 1 (8 parallel agents, 9 RTL modules): ibex_pmp.sv, ibex_cs_registers.sv,
    ibex_csr.sv, ibex_counter.sv, ibex_dummy_instr.sv, ibex_alu.sv,
    ibex_multdiv_fast.sv, ibex_load_store_unit.sv, ibex_branch_predict.sv.
    Produced 40 build_*_stream functions (see _original_unfixed_constrained_llm_l11.py).
  Independent review pass: read all 40 functions, cross-checked every RTL
    citation and op/csr-bucket index against the RTL and codec_l11.py.
    21 of 40 functions had a confirmed issue (see _review_report_llm_crt.md).
  Pass 2 (5 parallel agents, 5 more RTL scopes): ibex_core.sv (2 scopes: core_a
    = top-level glue muxes, core_b = illegal-insn/fcov/ECC wiring),
    ibex_decoder.sv, ibex_compressed_decoder.sv, ibex_icache.sv. Produced 24
    more build_*_stream functions in standalone draft files.
  Fix+merge pass (this file): every review finding independently re-verified
    against the actual RTL (not taken on the review's word), fixed in place;
    the 24 draft functions merged in after their own independent spot-check.
  Pass 3 (2 parallel isolated synthesis agents, targeting the specific bins
    still uncovered after passes 1-2's 63 streams plateaued at ~80% toggle
    regardless of instruction budget): ibex_core exception/trap-capture CSRs
    (mepc/mtvec/mtval/nt_branch_addr) + ibex_decoder bt_a_mux_sel_o. Each
    agent given ONLY the target RTL file + the exact uncovered-signal list +
    this project's own action-space description -- no conversation context,
    same "zero riscv-dv knowledge" constraint as passes 1-2. Independent
    isolated review pass verified every RTL/codec-index claim in both
    drafts (csr_bucket placeholders resolved to real L11_CSRS indices,
    caught an RVC prime-register bug in the nt_branch_addr stream (C_BEQZ's
    3-bit field can't address x0), confirmed all 11 "unreachable" findings).
    9 of 11 originally-targeted ibex_decoder gaps and 2 of 7 ibex_core gaps
    (csr_depc, debug_cause) confirmed structurally unreachable on this
    codec/build -- see the per-function docstrings below for exact RTL
    citations; no streams were fabricated for those.

FIX METHODOLOGY APPLIED (see individual docstrings for per-function detail):
  - "CSRRW/CSRRS/CSRRC with an unconstrained register never guarantees the
    claimed CSR byte" -> switched to CSRRWI/CSRRSI/CSRRCI (op=66/67/68) with a
    literal `rs1` field, which the codec treats as the 5-bit uimm directly
    (codec_l11.py:_encode_csri_l11, `uimm = rs1 & 0x1F`) -- NOT a register
    index -- so a literal there guarantees the exact bits written/set/cleared,
    for any target value/bitmask that fits in 5 bits (0-31). Where the target
    needs a bit beyond position 4 (PMP lock bit7, dummy_instr_mask bit5), a
    register was loaded via ADDI with one of the 5 fixed immediates
    (-2048,-100,0,100,2047 -> low byte 0x00,0x9C,0x00,0x64,0xFF) and used as
    the CSRRW/CSRRS source instead; where even that could not hit the exact
    literal originally claimed, the docstring says exactly what is and is not
    guaranteed rather than papering over it.
  - "x0/uimm=0 silently no-ops": ibex_decoder.sv:196-201 forces
    CSRRSI/CSRRCI with uimm==0 to CSR_OP_READ (no set/clear happens) --
    confirmed directly in the decoder. Fixed by using a nonzero uimm.
  - "Loop variable never wired into the instruction stream": wired in.
  - "Wrong op sample pool, low hit rate on stated target": narrowed to just
    the ops that actually assert the target signal, re-derived from the RTL
    condition and cross-checked against codec_l9.py's Op enum.
  - "Docstring/RTL-citation inaccuracy, code itself fine": docstring-only fix.
  - build_csr_shadow_stream: DELETED. Confirmed directly (ibex_core.sv:182,
    `localparam bit ShadowCSR = 1'b0;`) that ShadowCSR is a hardcoded
    localparam, not exposed as a module parameter -- gen_shadow in
    ibex_csr.sv is never elaborated in ANY Ibex build, not just this one.
    There is no RTL there to cover.

Op-code index reference (codec_l11.py):
  ALU:       0-18 (ADD/SUB/XOR/SLT/SLTU/OR/AND/SLL/SRL/SRA/ADDI/XORI/SLTI/SLTIU/ORI/ANDI/SLLI/SRLI/SRAI)
             87-142 (SH1ADD/SH2ADD/SH3ADD/ANDN/ORN/XNOR/ROL/ROR/MIN/MAX/MINU/MAXU/
                     SEXT_B/SEXT_H/ZEXT_H/REV8/ORC_B/CLZ/CTZ/CPOP/RORI/
                     BCLR/BSET/BINV/BEXT + legacy RV32B zbp/zbc/zbe/zbf)
  LOAD_STORE:19-26 (LB/LH/LW/LBU/LHU/SB/SH/SW)
  CSR:       27-29,66-68 (CSRRW/CSRRS/CSRRC/CSRRWI/CSRRSI/CSRRCI)
  MULDIV:    30-37 (MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU)
  BRANCH:    38-43 (BEQ/BNE/BLT/BGE/BLTU/BGEU)
  JUMP:      44,65 (JAL/JALR)
  COMPRESSED:45-60,73-82
  UPPER_IMM: 61,64 (LUI/AUIPC)
  SYSTEM:    62,63,69-72 (ECALL/EBREAK/FENCE/MRET/WFI/FENCE_I) -- CORRECTED:
             the pass-1 header claimed "WFI/MRET/SRET/DRET" for 69-72; verified
             against codec_l9.py's Op enum, the real mapping is
             69=FENCE, 70=MRET, 71=WFI, 72=FENCE_I. Ibex has no S-mode and no
             exposed DRET op, so SRET/DRET never existed in this codec. This
             was a documentation-only bug in the pass-1 header; every function
             that actually uses ops 69-72 already used the correct numeric
             index.
  EXCEPTION: 83-86 (illegal-instr / misaligned load-store, fixed encodings)

imm_bucket → immediate: {0: -2048, 1: -100, 2: 0, 3: 100, 4: 2047}
SHAMT_BUCKET_VALUES (ops 107, 112-136 -- I-type shamt group): {0:0, 1:8, 2:16, 3:24, 4:31}
"""

# ---------------------------------------------------------------------------
# CATEGORY_WEIGHTS_LLM — derived from RTL module complexity + coverage gap data
#
# Largest gaps (from prior RL run): ibex_counter (339 bins), ibex_core (296),
# ibex_cs_registers (191), ibex_pmp (118).
# CSR ops drive all four; system ops exercise the exception save/restore path;
# muldiv drives ibex_multdiv_fast FSM; load_store exercises ibex_load_store_unit.
# Re-examined during the fix+merge pass and left unchanged: still an accurate
# reflection of RTL complexity per category, no RTL evidence found to revise it.
# ---------------------------------------------------------------------------
CATEGORY_WEIGHTS_LLM: dict[str, float] = {
    "csr":        0.22,  # ibex_cs_registers large read-mux, ibex_counter writes,
                         # ibex_pmp cfg/addr writes, ibex_dummy_instr cpuctrl/secureseed
    "system":     0.14,  # ECALL/MRET: mepc_en/mcause_en/mstatus save+restore in ibex_cs_registers
    "alu":        0.18,  # ibex_alu: sign-mismatch comparator, RV32B ROL/ROR/BFP/SH*ADD
    "muldiv":     0.10,  # ibex_multdiv_fast: 7-state FSM, signed abs, div-by-zero
    "load_store": 0.12,  # ibex_load_store_unit: size mux, sign-ext, byte-enable, alignment
    "branch":     0.09,  # ibex_branch_predict: taken/not-taken, instr_b_taken toggle
    "compressed": 0.07,  # ibex_compressed_decoder: CI/CR/CA/CB/CIW/CL/CS format arms
    "jump":       0.04,  # ibex_branch_predict: instr_j always-taken, JALR
    "upper_imm":  0.03,  # LUI/AUIPC — minimal structural paths
    "exception":  0.01,  # illegal/misaligned: ibex_controller fault path
}

assert abs(sum(CATEGORY_WEIGHTS_LLM.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# ibex_pmp.sv — PMP mode case arms + permission check + lock + NAPOT + TOR
#
# pmp_cfg_t layout confirmed directly (ibex_pkg.sv: `typedef struct packed {
# logic lock; pmp_cfg_mode_e mode; logic exec; logic write; logic read; }
# pmp_cfg_t;`, and ibex_cs_registers.sv:1136 zero-pads it into an 8-bit CSR
# byte as `{lock, 2'b00, mode, exec, write, read}`): bit7=L, bits[6:5]=RES,
# bits[4:3]=mode (OFF=00,TOR=01,NA4=10,NAPOT=11), bit2=X, bit1=W, bit0=R.
# ---------------------------------------------------------------------------

def build_pmp_mode_sweep_stream(rng):
    """Target ibex_pmp.sv: region_match_all always_comb unique case arms.

    ibex_pmp.sv always_comb block:
      unique case (csr_pmp_cfg_i[r].mode)
        PMP_MODE_OFF   (2'b00): region_match_all = 1'b0
        PMP_MODE_NA4   (2'b10): region_match_all = region_match_eq
        PMP_MODE_NAPOT (2'b11): region_match_all = region_match_eq (with mask)
        PMP_MODE_TOR   (2'b01): region_match_all = (eq | gt) & lt

    PMP cfg byte (verified, see module header): OFF=0x00, TOR=0x08, NA4=0x10,
    NAPOT=0x18 (mode bits only, no perms/lock -- M-mode bypasses the PMP
    perm-check whenever L=0 and MSECCFG.MML=0, per ibex_pmp.sv's
    perm_check_wrapper, so leaving R/W/X clear here doesn't fault M-mode
    accesses).

    CORRECTED after review: the original CSRRW wrote pmpcfg from an
    unconstrained register (rs1=rng.randint(1,31)), so the mode byte was
    never actually guaranteed -- the mode sweep this function is named for
    was only hit by chance (~25% per mode). Switched to CSRRWI (op=66),
    whose `rs1` field the codec treats as a literal 5-bit uimm
    (codec_l11.py:_encode_csri_l11), not a register index -- all 4 mode
    bytes (0x00/0x08/0x10/0x18) fit in 5 bits, so each write now guarantees
    the exact mode requested for the CSR's region-0 byte (the other 3
    regions packed into the same pmpcfgN CSR get zeroed by the same write,
    which is fine: this function's job is a mode sweep, not a multi-region
    interaction test -- that's build_pmp_random_walk_stream's job).
    """
    stream = []
    CSRRWI = 66
    mode_bytes = [0x00, 0x08, 0x10, 0x18]  # OFF, TOR, NA4, NAPOT
    for bucket in range(37, 53):  # pmpaddr0..15 -- content need not be a
        # specific value here; TOR/NAPOT boundary construction is the
        # dedicated job of build_pmp_tor_boundary_stream / build_pmp_napot_stream.
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
    for bucket in range(33, 37):  # pmpcfg0..3 — sweep all 4 modes, guaranteed
        for mode_byte in mode_bytes:
            stream.append((CSRRWI, rng.randint(1, 31), mode_byte, 0, 2, bucket))
    for _ in range(8):
        op = rng.choice([19, 20, 21, 22, 23, 24, 25, 26])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(0, 4), 0))
    return stream


def build_pmp_permission_check_stream(rng):
    """Target ibex_pmp.sv: region_basic_perm_check — R/W/X bit combinations.

    ibex_pmp.sv:
      region_basic_perm_check[c][r] =
        ((pmp_req_type == PMP_ACC_EXEC)  & csr_pmp_cfg_i[r].exec)  |
        ((pmp_req_type == PMP_ACC_WRITE) & csr_pmp_cfg_i[r].write) |
        ((pmp_req_type == PMP_ACC_READ)  & csr_pmp_cfg_i[r].read)

    cfg byte: bit0=R, bit1=W, bit2=X → 0x01(R), 0x02(W), 0x04(X), 0x07(RWX).

    CORRECTED after review: same root cause as build_pmp_mode_sweep_stream —
    CSRRW from an unconstrained register never guaranteed these bit patterns.
    Switched to CSRRWI (literal uimm), which guarantees the exact perm byte
    (all 4 values fit in 5 bits) on every write.
    """
    stream = []
    CSRRWI = 66
    perm_bytes = [0x01, 0x02, 0x04, 0x07]  # R, W, X, RWX
    for bucket in range(33, 37):
        for perm_byte in perm_bytes:
            stream.append((CSRRWI, rng.randint(1, 31), perm_byte, 0, 2, bucket))
    for _ in range(6):
        stream.append((rng.choice([19, 20, 21, 22, 23]),
                       rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(0, 4), 0))
    return stream


def build_pmp_lock_stream(rng):
    """Target ibex_pmp.sv: L (lock) bit in pmpcfg — perm_check_wrapper locked path.

    ibex_pmp.sv perm_check_wrapper:
      When L=1 AND MML=0: M-mode access uses orig_perm_check (not bypass).
      When L=0 AND MML=0: M-mode always passes (access_fault=0 for M-mode).

    CORRECTED after review: the original CSRRW wrote pmpcfg from an
    unconstrained register, so L (bit7) was never actually guaranteed to be
    set — the locked-region path this function is named for was only hit by
    chance.

    PARTIAL FIX, honestly documented: L (bit7) cannot be reached by a
    CSRRWI literal the way the mode-only/perm-only writes above can (uimm is
    only 5 bits, i.e. bits[4:0] — bit7 is structurally unreachable via any
    CSRRWI/CSRRSI/CSRRCI immediate). Instead we load a GP register via ADDI
    with one of this action space's 5 fixed immediates and use it as the
    CSRRW source: imm_bucket=1 (-100 == 0xFFFFFF9C, low byte 0x9C =
    L=1,mode=NAPOT,X=1,W=0,R=0) and imm_bucket=4 (2047 == 0x000007FF, low
    byte 0xFF = L=1,mode=NAPOT,X=1,W=1,R=1). This guarantees L=1 with two
    different NAPOT permission combinations (X-only vs RWX) — it does NOT
    hit the exact TOR-mode+lock combos the original docstring claimed
    (0x88/0x9F), because none of this action space's 5 fixed ADDI
    immediates happen to produce a low byte with mode bits[4:3]=TOR(01) AND
    bit7=1 simultaneously. Guaranteeing what's guaranteeable (L=1, exercising
    the orig_perm_check path) rather than faking the unreachable exact byte.
    """
    stream = []
    ADDI = 10
    CSRRW = 27
    for bucket in range(37, 41):  # pmpaddr0..3
        stream.append((CSRRW, rng.randint(1, 31), rng.randint(1, 31), 0, 3, bucket))
    lock_x_reg = rng.randint(1, 31)
    lock_rwx_reg = rng.randint(1, 31)
    stream.append((ADDI, lock_x_reg, 0, 0, 1, 0))     # -100 -> 0x9C: L=1,NAPOT,X=1
    stream.append((ADDI, lock_rwx_reg, 0, 0, 4, 0))   # 2047 -> 0xFF: L=1,NAPOT,RWX=1
    stream.append((CSRRW, rng.randint(1, 31), lock_x_reg, 0, 2, 33))    # pmpcfg0: L=1, X-only
    stream.append((CSRRW, rng.randint(1, 31), lock_rwx_reg, 0, 2, 34))  # pmpcfg1: L=1, RWX
    for _ in range(4):
        op = rng.choice([19, 20, 21, 24, 25, 26])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(0, 4), 0))
    return stream


def build_pmp_tor_boundary_stream(rng):
    """Target ibex_pmp.sv: region_match_lt/gt comparators for TOR mode.

    ibex_pmp.sv TOR case arm:
      region_match_gt[r] = (fetch_pc_i[33:2] > csr_pmp_addr_i[r-1][33:2])
      region_match_lt[r] = (fetch_pc_i[33:2] < csr_pmp_addr_i[r][33:2])
      region_match_all = region_match_gt & region_match_lt

    CORRECTED after review: the pmpcfg write used CSRRW with imm_bucket=3
    controlling nothing about the CSR value (that field is ignored by
    CSRRW's encoding) and rs1=an unconstrained register — TOR mode was only
    ~25% likely to actually land. Switched to CSRRWI with uimm=0x0F
    (TOR mode + R/W/X all set, so subsequent loads/stores are actually
    permitted and exercise the comparators instead of merely being
    M-mode-bypassed), guaranteeing TOR mode on every run of this stream.
    """
    stream = []
    CSRRWI = 66
    stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 37))  # pmpaddr0 (lower bound)
    stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, 4, 38))  # pmpaddr1 (upper bound)
    stream.append((CSRRWI, rng.randint(1, 31), 0x0F, 0, 2, 33))  # pmpcfg0: TOR + RWX, guaranteed
    for _ in range(6):
        op = rng.choice([19, 20, 21, 22, 23, 24, 25, 26])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(0, 4), 0))
    return stream


def build_pmp_napot_stream(rng):
    """Target ibex_pmp.sv: NAPOT mask decode — region_match_eq with mask.

    ibex_pmp.sv NAPOT arm:
      mask = {1'b1, csr_pmp_addr_i[r][31:2] | ~({30{1'b1}} >> (size-3))}
      region_match_eq = (fetch_pc_i[33:2] & mask) == (addr & mask)

    CORRECTED after review: pmpcfg write used an unconstrained register, so
    NAPOT mode (vs OFF/NA4/TOR) was only ~25% likely. Switched to CSRRWI
    with uimm=0x1F (NAPOT mode + RWX), guaranteeing NAPOT mode + permitted
    access on every write. pmpaddr low-bit mask content is still whatever a
    prior instruction left in the source register (varied, but not a
    constructed 2^n-1 pattern) — a true systematic granule-width sweep would
    need an ADDI+shift sequence per width, which is a bigger change than
    this fix pass's scope; documented honestly rather than claimed.
    """
    stream = []
    CSRRWI = 66
    for bucket in range(37, 45):  # pmpaddr0..7
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0,
                       rng.randint(2, 4), bucket))
    for bucket in range(33, 37):  # pmpcfg0..3 — write NAPOT mode, guaranteed
        stream.append((CSRRWI, rng.randint(1, 31), 0x1F, 0, 2, bucket))
    for _ in range(4):
        op = rng.choice([19, 21, 22])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(0, 4), 0))
    return stream


def build_pmp_csrr_variants_stream(rng):
    """Target ibex_pmp.sv + ibex_cs_registers.sv: all 6 CSR access variants on PMP regs.

    Uses CSRRW(27)/CSRRS(28)/CSRRC(29)/CSRRWI(66)/CSRRSI(67)/CSRRCI(68) on
    pmpcfg0-3 (buckets 33-36) and pmpaddr0-15 (buckets 37-52) to exercise
    the write-enable, set-bits, clear-bits paths in ibex_cs_registers.sv for
    PMP CSR addresses, and to toggle pmp_cfg_we/pmp_addr_we strobes.
    """
    stream = []
    csr_ops = [27, 28, 29, 66, 67, 68]
    for bucket in list(range(33, 37)) + list(range(37, 53)):
        op = rng.choice(csr_ops)
        rd = rng.randint(1, 31)
        rs1 = rng.randint(0, 31)
        stream.append((op, rd, rs1, 0, 2, bucket))
    return stream


def build_pmp_random_walk_stream(rng):
    """Target ibex_pmp.sv: randomised cfg+addr writes followed by mixed accesses.

    Covers corner-case interactions between multiple PMP regions by writing
    all 4 pmpcfg registers and all 16 pmpaddr registers in random order with
    random values, then interleaving loads and stores so multiple region
    comparators fire simultaneously.
    """
    stream = []
    buckets = list(range(33, 53))  # pmpcfg0-3 + pmpaddr0-15
    rng.shuffle(buckets)
    for bucket in buckets:
        op = rng.choice([27, 28, 29])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
    for _ in range(12):
        op = rng.choice([19, 20, 21, 22, 23, 24, 25, 26])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(0, 4), 0))
    return stream


def build_pmp_all_stream(rng):
    """Combined PMP stream: mode sweep → lock → NAPOT → TOR → random walk.

    Chains all PMP sub-sequences to maximise single-run coverage of all
    ibex_pmp.sv always_comb branches in one long stream.
    """
    stream = []
    stream += build_pmp_mode_sweep_stream(rng)
    stream += build_pmp_lock_stream(rng)
    stream += build_pmp_napot_stream(rng)
    stream += build_pmp_tor_boundary_stream(rng)
    stream += build_pmp_permission_check_stream(rng)
    return stream


# ---------------------------------------------------------------------------
# ibex_cs_registers.sv — CSR read mux + exception save/restore + mstatus fields
# ---------------------------------------------------------------------------

def build_csr_full_sweep_stream(rng):
    """Target ibex_cs_registers.sv: the large always_comb CSR read mux.

    Doing a deterministic round-robin read (CSRRS rd, csr, x0) over all 70
    buckets ensures every case arm fires at least once. Alternates pure reads
    with read+writes to toggle both the read-mux output and the write path.
    """
    stream = []
    for bucket in range(70):
        if bucket % 3 == 0:
            stream.append((28, rng.randint(1, 31), 0, 0, 2, bucket))
        elif bucket % 3 == 1:
            stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0,
                           rng.randint(2, 4), bucket))
        else:
            stream.append((29, rng.randint(1, 31), rng.randint(1, 31), 0,
                           rng.randint(2, 4), bucket))
    return stream


def build_exception_save_restore_stream(rng):
    """Target ibex_cs_registers.sv: mepc_en/mcause_en/mstatus save+restore.

    On csr_save_cause_i (ECALL=62): mepc_q<=exception_pc, mcause_q<=cause,
      mstatus_q.mpie<=mie, mstatus_q.mie<=0.
    On csr_restore_mret_i (MRET=70): mstatus_q.mie<=mpie, mpie<=1.
    Sequence: set up mtvec/mie/mstatus → ECALL → read saved regs → MRET, repeat.
    """
    stream = []
    stream.append((27, 1, 1, 0, 3, 29))   # CSRRW mstatus
    stream.append((27, 1, 1, 0, 4, 30))   # CSRRW mie
    stream.append((27, 1, 1, 0, 3, 31))   # CSRRW mtvec
    stream.append((62, 0, 0, 0, 2, 0))    # ECALL → save path
    stream.append((28, rng.randint(1, 31), 0, 0, 2, 1))  # CSRRS mepc
    stream.append((28, rng.randint(1, 31), 0, 0, 2, 2))  # CSRRS mcause
    stream.append((70, 0, 0, 0, 2, 0))    # MRET → restore path
    for _ in range(3):
        stream.append((62, 0, 0, 0, 2, 0))
        stream.append((70, 0, 0, 0, 2, 0))
    return stream


def build_mstatus_bits_stream(rng):
    """Target ibex_cs_registers.sv: mstatus_q fields mie/mpie/mpp/mprv/tw.

    CSR_MSTATUS_MIE_BIT=3, MPIE_BIT=7, MPP_BIT=11:12, MPRV_BIT=17, TW_BIT=21
    (verified against ibex_pkg.sv parameters directly).
    Alternates CSRRS (set bits) and CSRRC (clear bits) to toggle each field.

    CORRECTED after review: `CSRRSI rd,x0,0,...` / `CSRRCI rd,x0,0,...`
    (rs1=0) pass uimm=0 (codec_l11.py:_encode_csri_l11, `uimm = rs1 & 0x1F`).
    ibex_decoder.sv:196-201 forces any CSRRS/CSRRC-class op (register or
    immediate form) with a zero set/clear operand to CSR_OP_READ — a pure
    read, no bit is actually set or cleared. The two instructions explicitly
    commented "set bit"/"clear bit" were silent no-ops. Fixed by using
    uimm=1<<3 (MIE bit), a nonzero literal that survives the decoder's
    zero-check and actually toggles CSR_MSTATUS_MIE_BIT.
    """
    stream = []
    mstatus_bucket = 29
    MIE_BIT = 1 << 3  # CSR_MSTATUS_MIE_BIT = 3
    stream.append((67, 0, MIE_BIT, 0, 2, mstatus_bucket))   # CSRRSI: set MIE (nonzero uimm)
    stream.append((68, 0, MIE_BIT, 0, 2, mstatus_bucket))   # CSRRCI: clear MIE (nonzero uimm)
    for _ in range(4):
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0,
                       rng.choice([2, 3, 4]), mstatus_bucket))
    stream.append((28, rng.randint(1, 31), 0, 0, 2, mstatus_bucket))
    return stream


# ---------------------------------------------------------------------------
# ibex_csr.sv — write-enable toggle + reset path
#
# build_csr_shadow_stream (pass 1) DELETED here: confirmed directly against
# ibex_core.sv:182 (`localparam bit ShadowCSR = 1'b0;`) that ShadowCSR is a
# hardcoded localparam, not forwarded as a configurable module parameter —
# ibex_csr.sv's gen_shadow block (shadow_q register, rd_error_o comparator)
# is therefore never elaborated in ANY Ibex build. There is no RTL there for
# any instruction stream to cover; the function's entire premise was
# unreachable, so it was removed rather than re-targeted or left as dead code.
# ---------------------------------------------------------------------------

def build_ibex_csr_wr_en_stream(rng):
    """Target ibex_csr.sv: wr_en_i gating on rdata_q register update.

    ibex_csr.sv always_ff:
      if (wr_en_i) rdata_q <= wr_data_i;  ← only updates when wr_en_i=1

    Toggle wr_en_i high then low by alternating CSRRW (writes, wr_en_i=1)
    and CSRRS x0,csr,x0 pure reads (wr_en_i=0, no side-effect) on the same CSR.
    Uses mscratch (bucket=0), mepc (bucket=1), mcause (bucket=2), mcycle
    (bucket=9), minstret (bucket=10) — plain registers with no write-side-
    effects that could trap execution.

    CORRECTED after review: the loop previously also iterated buckets 29/30/31
    (mstatus/mie/mtvec — verified against L11_CSRS[29]=0x300, [30]=0x304,
    [31]=0x305), directly contradicting this docstring's own safety claim.
    Writing an unconstrained value into mtvec (ibex_cs_registers.sv:617-621)
    relocates the trap base to a garbage address; if any exception/interrupt
    fires later in the program, the core jumps to that garbage PC. Removed
    29/30/31 from the bucket list so the code now matches what the docstring
    always claimed.
    """
    stream = []
    for bucket in [0, 1, 2, 9, 10]:
        # Write (wr_en_i = 1)
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
        # Pure read (wr_en_i = 0 — rs1=x0, no write for CSRRS)
        stream.append((28, rng.randint(1, 31), 0, 0, 2, bucket))
    return stream


def build_ibex_csr_reset_path_stream(rng):
    """Target ibex_csr.sv: write alternating 0/nonzero to maximise rdata_q transitions.

    ibex_csr.sv ff:
      rdata_q resets to 0 on rst_ni=0; updated to wr_data_i when wr_en_i=1.
    We cannot trigger a real reset from the instruction stream, but we can
    toggle bits in rdata_q by writing a mostly-nonzero pattern, then 0, then
    a literal-nonzero pattern, ensuring 0<->1 toggle coverage on the
    register's storage bits.
    Uses csrs that accept arbitrary writes: mscratch(0), mcause(2),
    mstatus(29), mtvec(31), cpuctrl(68).

    CORRECTED after review: the old comment ("Write nonzero (all-ones from
    large immediate via rs1 holding prior ALU result)") implied a
    deliberately-constructed all-ones register value, but no such ALU
    sequence existed anywhere in the function — rs1 was just
    rng.randint(1,31), an arbitrary, unconstrained register. Now uses ADDI
    imm_bucket=1 (-100 == 0xFFFFFF9C — bits[31:8] are all 1 via sign
    extension) to guarantee a mostly-all-ones write, alternated with a
    literal all-zero write (rs1=x0, still guaranteed) and a CSRRWI with
    uimm=0x1F (bits[4:0] guaranteed all-1, upper bits guaranteed zero) —
    together these guarantee toggle coverage on both the high
    sign-extension-derived bits and the low literal-immediate bits, though
    NOT a single write that is all-1s in every one of the 32 bits (that
    exact pattern is unreachable from this 5-fixed-immediate/5-bit-uimm
    action space without a dedicated LUI+ORI sequence, which is out of this
    fix pass's scope).
    """
    stream = []
    ADDI = 10
    CSRRWI = 66
    for bucket in [0, 2, 29, 31, 68]:
        allones_reg = rng.randint(1, 31)
        stream.append((ADDI, allones_reg, 0, 0, 1, 0))                     # -100 -> mostly-1s pattern (guaranteed)
        stream.append((27, rng.randint(1, 31), allones_reg, 0, 2, bucket))  # write it
        stream.append((27, rng.randint(1, 31), 0, 0, 2, bucket))            # write zero (rs1=x0, guaranteed)
        stream.append((CSRRWI, rng.randint(1, 31), 0x1F, 0, 2, bucket))     # literal low-5-bits-all-1 (guaranteed)
        stream.append((28, rng.randint(1, 31), 0, 0, 2, bucket))            # read back
    return stream


# ---------------------------------------------------------------------------
# ibex_counter.sv — counter write + high-half + increment + inhibit + HPM events
# ---------------------------------------------------------------------------

def build_counter_write_stream(rng):
    """Target ibex_counter.sv: counter_we_i (low half) vs counterh_we_i (high half).

    ibex_counter.sv always_comb:
      if (counterh_we_i): counter_load[63:32] = counter_val_i  ← high arm
      else:               counter_load[31:0]  = counter_val_i  ← low arm

    Low half CSRs: mcycle(9), minstret(10), mhpmcounter3-8(14-19),
    mhpmcounter9-14(56-61). High half CSRs: mcycleh(11), minstreth(12).

    CORRECTED after review (docstring only, code was fine): the old
    docstring labeled buckets 56-61 as "mhpmcounter3-8" again — verified
    against L11_CSRS directly (printed the list): buckets 14-19 are indeed
    0xB03-0xB08 (mhpmcounter3..8), but 56-61 are 0xB09-0xB0E
    (mhpmcounter9..14) — a distinct set of counter instances, not a
    duplicate of 3-8. The code itself correctly drives counter_we_i on both
    groups; only the label was wrong.
    """
    stream = []
    low_buckets = [9, 10] + list(range(14, 20)) + list(range(56, 62))
    high_buckets = [11, 12]
    for bucket in low_buckets:
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
    for bucket in high_buckets:
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
    return stream


def build_counter_write_high_stream(rng):
    """Target ibex_counter.sv: counterh_we_i path specifically.

    ibex_counter.sv counterh_we_i arm:
      counter_load[63:32] = counter_val_i;
      counter_load[31:0]  = counter[31:0];  ← preserves low half

    mcycleh = bucket 11 (0xb80), minstreth = bucket 12 (0xb82).
    Alternates writes to low then high to verify both halves update correctly.
    """
    stream = []
    for _ in range(6):
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), 9))   # mcycle
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), 11))  # mcycleh
        stream.append((28, rng.randint(1, 31), 0, 0, 2, 9))   # read mcycle
        stream.append((28, rng.randint(1, 31), 0, 0, 2, 11))  # read mcycleh
    return stream


def build_counter_inc_stream(rng):
    """Target ibex_counter.sv: counter_d = counter_upd (increment path).

    ibex_counter.sv:
      counter_upd = counter[63:0] + {{63{1'b0}}, counter_inc_i}
      counter_d = we ? counter_load : counter_upd  ← increment when !we

    The cycle counter increments every cycle; instret every retired instruction.
    Running a burst of ALU instructions (no stalls) maximises the toggle rate
    on the adder carry chain and on counter_inc_i (always 1 unless inhibited).
    Read mcycle before and after to verify increment happened.
    """
    stream = []
    stream.append((28, rng.randint(1, 31), 0, 0, 2, 9))  # read mcycle baseline
    for _ in range(16):
        op = rng.choice([0, 1, 5, 6, 10, 14, 15])  # fast ALU ops (1-cycle)
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    stream.append((28, rng.randint(1, 31), 0, 0, 2, 9))  # read mcycle after
    stream.append((28, rng.randint(1, 31), 0, 0, 2, 10)) # read minstret
    return stream


def build_counter_inhibit_stream(rng):
    """Target ibex_counter.sv + ibex_cs_registers.sv: mcountinhibit freeze/unfreeze.

    mcountinhibit (CSR 0x320, bucket=13): confirmed directly
    (ibex_cs_registers.sv:1382-1383, `.counter_inc_i(mhpmcounter_incr[0] &
    ~mcountinhibit[0])`) that bit0 gates the cycle counter's counter_inc_i.
    Writing bit0=1 freezes mcycle, toggling the counter_d = counter_upd vs
    counter[63:0] mux in ibex_counter.sv's always_comb.

    CORRECTED after review: the original CSRRW wrote from an unconstrained
    register (rs1=rng.randint(1,31)), so "nonzero" (and specifically bit0)
    was never actually guaranteed. Switched to CSRRSI/CSRRCI with a literal
    uimm=1, guaranteeing bit0 set/cleared regardless of any other register's
    state.
    """
    stream = []
    inhibit_bucket = 13
    stream.append((67, 0, 1, 0, 2, inhibit_bucket))  # CSRRSI: set CY-inhibit bit0 (guaranteed)
    for _ in range(4):
        stream.append((10, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))  # ADDI
    stream.append((68, 0, 1, 0, 2, inhibit_bucket))  # CSRRCI: clear CY-inhibit bit0 (guaranteed)
    for _ in range(4):
        stream.append((10, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))  # ADDI
    stream.append((28, rng.randint(1, 31), 0, 0, 2, 9))   # CSRRS mcycle
    return stream


def build_counter_hpmcounter_stream(rng):
    """Target ibex_counter.sv instances for mhpmcounter3-14 + mhpmevent CSRs.

    ibex_cs_registers.sv instantiates ibex_counter for mhpmcounter3..14.
    Writing mhpmevent3-8 (buckets 20-25) configures which event increments which
    counter. Writing mhpmcounter3-8 (buckets 14-19) exercises the we path on
    those counter instances. Reading them back hits the read-mux arms for HPM CSRs.
    """
    stream = []
    event_buckets   = list(range(20, 26))   # mhpmevent3-8 (0x323..0x328)
    counter_buckets = list(range(14, 20))   # mhpmcounter3-8 (0xb03..0xb08)
    for bucket in event_buckets:
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
    for bucket in counter_buckets:
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
        stream.append((28, rng.randint(1, 31), 0, 0, 2, bucket))
    return stream


# ---------------------------------------------------------------------------
# ibex_dummy_instr.sv — LFSR seed + enable + mask + arm sweep + combined
#
# cpu_ctrl_sts_part_t layout confirmed directly (ibex_cs_registers.sv:206-213,
# packed struct MSB..LSB): bit7=double_fault_seen, bit6=sync_exc_seen,
# bits[5:3]=dummy_instr_mask, bit2=dummy_instr_en, bit1=data_ind_timing,
# bit0=icache_enable. CORRECTED after review: pass-1 claimed "cpuctrl
# bit0=dummy_instr_en" and "bits[3:1]=dummy_instr_mask" — both wrong by one
# field's width. dummy_instr_en is actually bit2 (weight 4); bit0 is
# icache_enable, an unrelated signal. Every function below that writes
# cpuctrl to "enable dummy_instr" was previously targeting the wrong bit
# entirely (independent of the separate unconstrained-register issue the
# review also flagged for these functions).
# ---------------------------------------------------------------------------

DUMMY_INSTR_EN_BIT = 1 << 2  # cpu_ctrl_sts_part_t.dummy_instr_en, weight 4


def build_dummy_instr_enable_stream(rng):
    """Target ibex_dummy_instr.sv: insert_dummy_instr path + DUMMY_* enum arms.

    insert_dummy_instr = dummy_instr_en_i & (dummy_cnt_q == dummy_cnt_threshold)
    unique case(lfsr_data.instr_type):
      DUMMY_ADD/MUL/DIV/AND — all 4 arms exercised as LFSR cycles.

    CORRECTED after review + independent bit-position check (see module
    header): dummy_instr_en is cpuctrl bit2 (weight 4), not bit0. The
    original CSRRW also wrote from rs1=1 (register x1's incidental prior
    content), never guaranteeing the bit was actually set. Switched to
    CSRRSI/CSRRCI with literal uimm=DUMMY_INSTR_EN_BIT, which set/clear
    exactly that bit and leave icache_enable/data_ind_timing/mask untouched.
    """
    stream = []
    cpuctrl_bucket = 68
    stream.append((67, 0, DUMMY_INSTR_EN_BIT, 0, 2, cpuctrl_bucket))  # CSRRSI: enable (guaranteed)
    for _ in range(20):
        op = rng.choice([0, 1, 2, 5, 6, 7, 8, 9])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    stream.append((68, 0, DUMMY_INSTR_EN_BIT, 0, 2, cpuctrl_bucket))  # CSRRCI: disable (guaranteed)
    return stream


def build_dummy_instr_seed_stream(rng):
    """Target ibex_dummy_instr.sv: dummy_instr_seed_d XOR path.

    dummy_instr_seed_d = dummy_instr_seed_q ^ dummy_instr_seed_i
    Asserted when secureseed (0x7C1, bucket=69) is written.
    Each write XORs into seed_q, changing LFSR state and which DUMMY_* arm
    fires — the exact XORed value doesn't need to be controlled for this to
    work, so secureseed writes are left register-sourced (varied) on purpose.

    CORRECTED after review + bit-position check: the "enable" line now uses
    CSRRSI with literal uimm=DUMMY_INSTR_EN_BIT (bit2), not a CSRRW from an
    unconstrained register targeting the wrong bit (bit0).
    """
    stream = []
    stream.append((67, 0, DUMMY_INSTR_EN_BIT, 0, 2, 68))  # CSRRSI: enable dummy_instr (bit2, guaranteed)
    for _ in range(8):
        stream.append((27, 0, rng.randint(1, 31), 0, rng.randint(2, 4), 69))  # secureseed
    for _ in range(10):
        stream.append((rng.choice([0, 8, 9, 10]), rng.randint(1, 31),
                       rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    return stream


def build_ibex_dummy_instr_lfsr_arm_sweep_stream(rng):
    """Target ibex_dummy_instr.sv: all 4 DUMMY_* case arms systematically.

    unique case (lfsr_data.instr_type):
      DUMMY_ADD (2'b00): dummy_set=7'b0000000, dummy_opcode=3'b000 → ADD
      DUMMY_MUL (2'b01): dummy_set=7'b0000001, dummy_opcode=3'b000 → MUL
      DUMMY_DIV (2'b10): dummy_set=7'b0000001, dummy_opcode=3'b100 → DIV
      DUMMY_AND (2'b11): dummy_set=7'b0000000, dummy_opcode=3'b111 → AND

    Run a long burst with dummy enabled. The LFSR's instr_type field cycles
    through 2-bit values so all 4 arms are reached within O(4) dummy insertions.
    Interleave secureseed writes to vary LFSR state and ensure the sweep doesn't
    get stuck in a degenerate LFSR sequence.

    CORRECTED after review + bit-position check: enable/disable now use
    CSRRSI/CSRRCI with literal uimm=DUMMY_INSTR_EN_BIT (bit2, guaranteed),
    not a CSRRW-from-unconstrained-register targeting bit0 (icache_enable).
    """
    stream = []
    stream.append((67, 0, DUMMY_INSTR_EN_BIT, 0, 2, 68))   # CSRRSI: enable (bit2, guaranteed)
    for i in range(32):
        stream.append((rng.choice([0, 1, 5, 6, 10, 14]), rng.randint(1, 31),
                       rng.randint(1, 31), rng.randint(1, 31), 2, 0))
        if i % 8 == 7:
            stream.append((27, 0, rng.randint(1, 31), 0, rng.randint(2, 4), 69))  # secureseed
    stream.append((68, 0, DUMMY_INSTR_EN_BIT, 0, 2, 68))   # CSRRCI: disable (guaranteed)
    return stream


def build_ibex_dummy_instr_mask_stream(rng):
    """Target ibex_dummy_instr.sv: dummy_cnt_threshold mask bits (dummy_instr_mask_i).

    dummy_cnt_threshold = lfsr_data.cnt & {dummy_instr_mask_i, {TIMEOUT_CNT_W-3{1'b1}}}

    CORRECTED after review + independent bit-position check: the original
    docstring claimed "cpuctrl bits[3:1] = dummy_instr_mask_i" and the code
    wrote via CSRRW with an unconstrained register (`rs1=rng.randint(1,31)`)
    whose value the loop's own `mask_val` counter was never wired into —
    dummy_instr_mask_i was not actually swept at all. Verified directly
    against ibex_cs_registers.sv's cpu_ctrl_sts_part_t (see module header):
    dummy_instr_mask actually lives at bits[5:3], not [3:1].

    PARTIAL FIX, honestly documented: CSRRSI/CSRRCI's uimm is only 5 bits
    (bits[4:0] of the value), so mask bit0 (cpuctrl bit3, weight 8) and mask
    bit1 (cpuctrl bit4, weight 16) ARE individually set/clear-able via a
    literal uimm — giving an EXACT, guaranteed sweep of
    dummy_instr_mask_i in {0,1,2,3}. Mask bit2 (cpuctrl bit5, weight 32) is
    NOT reachable by any CSRRxI immediate (bit5 is beyond the 5-bit uimm's
    range). For that top bit we fall back to a best-effort CSRRS with a
    register loaded via ADDI imm_bucket=4 (2047 -> low byte 0xFF, which does
    include bit5) — this ORs in the whole 0xFF byte, so it also forces
    icache_enable/data_ind_timing/dummy_instr_en/mask bits 0-1 all to 1
    simultaneously as a side effect; it guarantees mask bit2 gets exercised
    at least once, but is coarser than the exact 2-bit sweep above.
    """
    stream = []
    CSRRSI, CSRRCI = 67, 68
    ADDI = 10
    cpuctrl_bucket = 68
    # Ensure dummy_instr_en=1 up front (bit2, guaranteed literal).
    stream.append((CSRRSI, 0, DUMMY_INSTR_EN_BIT, 0, 2, cpuctrl_bucket))
    # Exact, guaranteed sweep of dummy_instr_mask_i in {0,1,2,3} via bit3/bit4.
    for mask_val in range(4):
        set_bits = ((mask_val & 1) << 3) | (((mask_val >> 1) & 1) << 4)
        clear_bits = (~set_bits) & 0b11000  # the mask bits NOT wanted this round
        if set_bits:
            stream.append((CSRRSI, 0, set_bits, 0, 2, cpuctrl_bucket))
        if clear_bits:
            stream.append((CSRRCI, 0, clear_bits, 0, 2, cpuctrl_bucket))
        for _ in range(6):
            stream.append((rng.choice([0, 10, 5]), rng.randint(1, 31),
                           rng.randint(1, 31), 0, 2, 0))
    # Best-effort hit on mask bit2 (weight 32, unreachable via any literal uimm).
    allones_reg = rng.randint(1, 31)
    stream.append((ADDI, allones_reg, 0, 0, 4, 0))     # 2047 -> low byte 0xFF (includes bit5)
    stream.append((28, 0, allones_reg, 0, 2, cpuctrl_bucket))  # CSRRS: OR 0xFF into cpuctrl
    for _ in range(6):
        stream.append((rng.choice([0, 10, 5]), rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))
    stream.append((CSRRCI, 0, 0b11111, 0, 2, cpuctrl_bucket))  # clear bits[4:0] -> disable cleanly
    return stream


def build_ibex_dummy_instr_combined_stream(rng):
    """Combined: seed→enable→arm_sweep→disable cycle, repeated with varying seeds.

    Chains seed writes, enable, ALU burst (to trigger insertions), disable.
    Exercises: dummy_instr_seed_d XOR path, insert_dummy_instr toggle,
    all 4 DUMMY_* instr_type case arms, lfsr_en toggle, dummy_cnt reset.

    CORRECTED after review + bit-position check: enable/disable now use
    CSRRSI/CSRRCI with literal uimm=DUMMY_INSTR_EN_BIT (bit2, guaranteed).
    """
    stream = []
    for _ in range(3):
        stream.append((27, 0, rng.randint(1, 31), 0, rng.randint(2, 4), 69))  # seed
        stream.append((67, 0, DUMMY_INSTR_EN_BIT, 0, 2, 68))                  # CSRRSI: enable (guaranteed)
        for _ in range(12):
            stream.append((rng.choice([0, 1, 5, 10]), rng.randint(1, 31),
                           rng.randint(1, 31), rng.randint(1, 31), 2, 0))
        stream.append((68, 0, DUMMY_INSTR_EN_BIT, 0, 2, 68))                  # CSRRCI: disable (guaranteed)
    return stream


# ---------------------------------------------------------------------------
# ibex_alu.sv — comparator sign-mismatch + multicycle shift + single-bit ops
# ---------------------------------------------------------------------------

def build_alu_compare_sign_mismatch_stream(rng):
    """Target ibex_alu.sv: is_greater_equal sign-mismatch branch.

    ibex_alu.sv always_comb is_greater_equal:
      if ((operand_a_i[31] ^ operand_b_i[31]) == 1'b0):
        is_greater_equal = (adder_result[31] == 1'b0)  // same-sign
      else:
        is_greater_equal = operand_a_i[31] ^ (cmp_signed)  // mixed-sign

    Mixed-sign: load -2048 (bucket=0) → bit[31]=1, load +2047 (bucket=4) → bit[31]=0.
    cmp_signed=1 for SLT/BGE/MIN/MAX; cmp_signed=0 for SLTU/BLTU/MINU/MAXU.
    """
    stream = []
    ADDI = 10
    SLT, SLTU = 3, 4
    MIN_OP, MAX_OP = 95, 96
    MINU_OP, MAXU_OP = 97, 98
    BGE, BLTU = 41, 42

    for _ in range(5):
        rd_neg = rng.randint(1, 15)
        rd_pos = rng.randint(16, 31)
        stream.append((ADDI, rd_neg, 0, 0, 0, 0))
        stream.append((ADDI, rd_pos, 0, 0, 4, 0))
        stream.append((SLT,   rng.randint(1, 31), rd_neg, rd_pos, 2, 0))
        stream.append((SLTU,  rng.randint(1, 31), rd_neg, rd_pos, 2, 0))
        stream.append((MIN_OP,  rng.randint(1, 31), rd_neg, rd_pos, 2, 0))
        stream.append((MINU_OP, rng.randint(1, 31), rd_neg, rd_pos, 2, 0))
        stream.append((BGE,  0, rd_neg, rd_pos, 1, 0))
        stream.append((BLTU, 0, rd_pos, rd_neg, 3, 0))
    return stream


def build_alu_rv32b_stream(rng):
    """Target ibex_alu.sv: RV32B multicycle ROL/ROR, single-bit ops, SH*ADD.

    ROL(93)/ROR(94): 2-cycle, shift_left toggles. imd_val_we_o[0] in cycle 1.
    BSET(109)/BCLR(108)/BINV(110): shift_sbmode=1 → shift_operand=32'h8000_0000.
    BEXT(111): shift_sbmode=0, result[0] extraction.
    SH1ADD(87)/SH2ADD(88)/SH3ADD(89): adder_op_a_shift1/2/3 mux arms.
    """
    stream = []
    for rs2 in [0, 1, 2, 15, 31]:
        stream.append((93, rng.randint(1, 31), rng.randint(1, 31), rs2, 2, 0))  # ROL
        stream.append((94, rng.randint(1, 31), rng.randint(1, 31), rs2, 2, 0))  # ROR
    stream.append((107, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))       # RORI
    for op in [109, 108, 110]:
        for bit_rs2 in [0, 1, 15, 31]:
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), bit_rs2, 2, 0))
    for bit_rs2 in [0, 1, 16, 31]:
        stream.append((111, rng.randint(1, 31), rng.randint(1, 31), bit_rs2, 2, 0))  # BEXT
    for op in [87, 88, 89]:
        for _ in range(3):
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    return stream


# ---------------------------------------------------------------------------
# ibex_multdiv_fast.sv — FSM states + signed abs + div-by-zero + MULH + interleaved
# ---------------------------------------------------------------------------

def build_multdiv_signed_fsm_stream(rng):
    """Target ibex_multdiv_fast.sv: MD_ABS_A, MD_ABS_B, MD_CHANGE_SIGN states.

    FSM: MD_IDLE → MD_ABS_A (neg numerator) → MD_ABS_B (neg denominator) →
         MD_COMP → MD_LAST → MD_CHANGE_SIGN (one neg operand) → MD_FINISH.
    DIV(34)/REM(36): signed, MD_ABS_* entered when operand[31]=1.
    ADDI with imm_bucket=0 (-2048) sets bit[31]=1 in rd.
    """
    stream = []
    ADDI = 10
    neg_regs = [rng.randint(1, 8) for _ in range(3)]
    pos_regs = [rng.randint(16, 24) for _ in range(3)]
    for r in neg_regs:
        stream.append((ADDI, r, 0, 0, 0, 0))
    for r in pos_regs:
        stream.append((ADDI, r, 0, 0, 4, 0))
    for nr in neg_regs:
        for pr in pos_regs:
            stream.append((34, rng.randint(1, 31), nr, pr, 2, 0))
    for pr in pos_regs:
        for nr in neg_regs:
            stream.append((34, rng.randint(1, 31), pr, nr, 2, 0))
    for nr1 in neg_regs[:2]:
        stream.append((34, rng.randint(1, 31), nr1, neg_regs[-1], 2, 0))
    for nr in neg_regs:
        stream.append((36, rng.randint(1, 31), nr, pos_regs[0], 2, 0))  # REM
    stream.append((35, rng.randint(1, 15), rng.randint(1, 31), 0, 2, 0))  # DIVU /0
    stream.append((37, rng.randint(1, 15), rng.randint(1, 31), 0, 2, 0))  # REMU /0
    return stream


def build_multdiv_mulh_stream(rng):
    """Target ibex_multdiv_fast.sv: MULH 2-cycle MULL→MULH FSM transition.

    gen_mult_single_cycle typedef: MULL, MULH.
    MULL state: if operator != MD_OP_MULL → mult_state_d = MULH (for MULH/MULHSU/MULHU).
    MULH state: computes ah*bh, mult_valid=1.
    MUL(30) stays MULL; MULH(31)/MULHSU(32)/MULHU(33) transition to MULH.
    """
    stream = []
    for _ in range(4):
        stream.append((30, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    for _ in range(6):
        stream.append((31, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    for _ in range(4):
        stream.append((32, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    for _ in range(4):
        stream.append((33, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    return stream


def build_ibex_multdiv_fast_div_by_zero_stream(rng):
    """Target ibex_multdiv_fast.sv: div_by_zero_q flip-flop.

    CORRECTED after review (docstring only, code was fine): verified
    directly against ibex_multdiv_fast.sv:434-437 — div_by_zero_d is not a
    standing combinational assignment; it's set only inside the MD_IDLE case
    arm: `div_by_zero_d = equal_to_zero_i;` (equal_to_zero_i is an
    ALU-computed zero-flag on the input operand, not literally
    `op_denominator_d == '0`), and only sampled once per divide (in
    MD_IDLE) — other states hold div_by_zero_q. The stream itself (using
    rs2=x0 for zero-denominator divides) is correct and does exercise the
    real signal.

    Using rs2=x0 (always zero) as denominator for DIV(34)/DIVU(35)/REM(36)/
    REMU(37) ensures equal_to_zero_i is asserted for the MD_IDLE sample, so
    div_by_zero_q=1. Mix with nonzero-denominator divides to toggle
    div_by_zero_q 0→1→0.
    """
    stream = []
    for _ in range(4):
        # Divide by nonzero first (div_by_zero_q = 0)
        stream.append((34, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
        # Divide by zero (rs2=x0, div_by_zero_q = 1)
        stream.append((34, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))
        stream.append((35, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))
        stream.append((36, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))
        stream.append((37, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))
    return stream


def build_ibex_multdiv_fast_interleaved_stream(rng):
    """Target ibex_multdiv_fast.sv: operator_i toggle between MD_OP_MULL and MD_OP_DIV.

    Interleaving MUL (MD_OP_MULL) and DIV (MD_OP_DIV) forces the FSM to reset
    through MD_IDLE between operations and toggles the operator_i signal.
    Also exercises MD_OP_MULH by inserting MULH between divides.
    """
    stream = []
    for _ in range(6):
        stream.append((30, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))  # MUL
        stream.append((34, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))  # DIV
        stream.append((31, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))  # MULH
        stream.append((35, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))  # DIVU
    return stream


# ---------------------------------------------------------------------------
# ibex_load_store_unit.sv — size mux + sign extension + byte-enable + misaligned
# ---------------------------------------------------------------------------

def build_load_store_size_stream(rng):
    """Target ibex_load_store_unit.sv: lsu_type_i size mux + lsu_sign_ext_i path.

    CORRECTED after review (docstring only, code was fine): verified
    directly against ibex_load_store_unit.sv:119
    (`unique case (lsu_type_i) // Data type 00 Word, 01 Half word, 11,10
    byte`) — lsu_type_i[1:0]: 2'b00=WORD, 2'b01=HALFWORD, 2'b10/2'b11=BYTE.
    The old docstring had this backwards (claimed 00=byte). Doesn't affect
    the code itself (the function enumerates by opcode, not by manually
    setting lsu_type_i bits), only the citation was wrong.
    lsu_sign_ext_i: 0 for LBU(22)/LHU(23), 1 for LB(19)/LH(20).
    data_be_o: byte-enable derived from size+addr[1:0].
    """
    stream = []
    load_ops  = [19, 20, 21, 22, 23]
    store_ops = [24, 25, 26]
    all_imm   = [0, 1, 2, 3, 4]
    for op in load_ops:
        for imm in all_imm:
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0, imm, 0))
    for op in store_ops:
        for imm in all_imm:
            stream.append((op, 0, rng.randint(1, 31), rng.randint(1, 31), imm, 0))
    return stream


def build_ibex_load_store_sign_ext_stream(rng):
    """Target ibex_load_store_unit.sv: sign-extend mux for byte/half loads.

    CORRECTED after review (docstring only, code was fine): the old
    docstring quoted signals (`rdata_b_i`, `rdata_h_i`) that don't exist
    anywhere in ibex_load_store_unit.sv — a fabricated paraphrase, not an
    actual transcription. Verified directly (lines 281-318): the real logic
    is a 4-way `case (rdata_offset_q)` on `data_rdata_i` byte/half slices,
    each arm gated by the REGISTERED `data_sign_ext_q` (captured from the
    unregistered lsu_sign_ext_i input on ctrl_update, not lsu_sign_ext_i
    directly):
      rdata_b_ext = data_sign_ext_q ? {{24{data_rdata_i[hi]}}, data_rdata_i[lo:hi-7]}
                                     : {24'h0, data_rdata_i[lo:hi-7]}
    (byte/halfword slice selected by rdata_offset_q; symmetric structure for
    halfwords). The instruction stream itself is unaffected: alternating
    LB/LH (signed, sets data_sign_ext_q=1) vs LBU/LHU (unsigned, sets
    data_sign_ext_q=0) still correctly toggles the mux on every retiring
    load.
    """
    stream = []
    signed_ops   = [19, 20]   # LB, LH → data_sign_ext_q = 1
    unsigned_ops = [22, 23]   # LBU, LHU → data_sign_ext_q = 0
    for _ in range(10):
        s_op = rng.choice(signed_ops)
        u_op = rng.choice(unsigned_ops)
        rs1 = rng.randint(1, 31)
        stream.append((s_op, rng.randint(1, 31), rs1, 0, rng.randint(0, 4), 0))
        stream.append((u_op, rng.randint(1, 31), rs1, 0, rng.randint(0, 4), 0))
    return stream


def build_ibex_load_store_misaligned_stream(rng):
    """Target ibex_load_store_unit.sv: misaligned exception + load_err_o/store_err_o.

    Exception ops 83-86 are fixed encodings for misaligned load/store (and illegal
    instructions). Emitting them exercises the alignment-check mux and the
    load_err_o/store_err_o output paths, which feed into ibex_controller's
    fault-handling state machine.
    Surround with normal loads/stores to toggle load_err_o 0→1→0.
    """
    stream = []
    for _ in range(8):
        stream.append((rng.choice([19, 20, 21]), rng.randint(1, 31),
                       rng.randint(1, 31), 0, rng.randint(0, 4), 0))  # normal load
        stream.append((rng.choice([83, 84, 85, 86]), 0, 0, 0, 2, 0))  # exception/misaligned
    return stream


def build_ibex_load_store_be_sweep_stream(rng):
    """Target ibex_load_store_unit.sv: data_be_o byte-enable multiplexer.

    CORRECTED after review (docstring only, code was fine): verified
    directly against ibex_load_store_unit.sv:118-138. The old docstring
    claimed "word → 4'b1111 always"; the real WORD arm is
    `data_offset`-dependent:
      2'b00: 4'b1111   2'b01: 4'b1110   2'b10: 4'b1100   2'b11: 4'b1000
    (1111 only when data_offset==2'b00). Separately, the "SB all offsets"
    framing was misleading: codec_l11.py's imm_bucket table
    ({0:-2048,1:-100,2:0,3:100,4:2047}) means buckets 0-3 all have
    imm[1:0]==00 (all four values are ≡0 mod 4), so the loop's imm choices
    don't actually vary addr[1:0] — any offset variation comes from the
    unconstrained rs1 register's low bits, not from the op/imm enumeration.
    The instruction mix itself (SB/SH/SW across varied rs1/rs2) still
    exercises the BE mux via whatever addr[1:0] the registers happen to
    produce; only the citation and framing were wrong.
    """
    stream = []
    for _ in range(4):
        for op, imm in [(24, 0), (24, 1), (24, 2), (24, 3),   # SB, varied rs1-derived offsets
                        (25, 0), (25, 2),                       # SH, varied rs1-derived offsets
                        (26, 0)]:                               # SW, varied rs1-derived offset
            rs1 = rng.randint(1, 31)
            rs2 = rng.randint(1, 31)
            stream.append((op, 0, rs1, rs2, imm, 0))
    for _ in range(4):
        for op, imm in [(19, 0), (19, 1), (20, 0), (21, 0)]:  # loads
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0, imm, 0))
    return stream


# ---------------------------------------------------------------------------
# ibex_branch_predict.sv — predict_taken toggle + case arms + compressed
#
# instr_cj/instr_cb confirmed directly (ibex_branch_predict.sv:75-76):
#   instr_cb = (instr[1:0]==2'b01) & (instr[15:13] in {3'b110,3'b111})
#   instr_cj = (instr[1:0]==2'b01) & (instr[15:13] in {3'b101,3'b001})
# Cross-checked against codec_l9.py's Op enum: only C_JAL(73, funct3=001)
# and C_J(74, funct3=101) satisfy instr_cj; only C_BEQZ(75) and C_BNEZ(76)
# satisfy instr_cb.
# ---------------------------------------------------------------------------

def build_branch_predict_stream(rng):
    """Target ibex_branch_predict.sv: predict_branch_taken_o toggle.

    instr_b_taken = (instr_b & imm_b_type[31]) | (instr_cb & imm_cb_type[31])
    predict_branch_taken_o = fetch_valid_i & (instr_j | instr_cj | instr_b_taken)

    Backward offset (imm_bucket=1, -100): imm_b_type[31]=1 → predicted taken.
    Forward offset (imm_bucket=3, +100): imm_b_type[31]=0 → predicted not-taken.
    JAL(44): instr_j=1 → always taken regardless of offset.
    """
    stream = []
    branch_ops = list(range(38, 44))
    for _ in range(12):
        op = rng.choice(branch_ops)
        stream.append((op, 0, rng.randint(0, 31), rng.randint(0, 31), rng.choice([1, 3]), 0))
    for _ in range(3):
        stream.append((44, rng.randint(1, 31), 0, 0, rng.choice([1, 3]), 0))
    for _ in range(4):
        stream.append((rng.choice(list(range(73, 83))), rng.randint(0, 31),
                       rng.randint(0, 31), 0, rng.choice([1, 3]), 0))
    return stream


def build_ibex_branch_predict_backward_branch_stream(rng):
    """Target ibex_branch_predict.sv: instr_b_taken=1 (backward branch taken).

    assign instr_b_taken = (instr_b & imm_b_type[31]) | ...
    imm_b_type[31]=1 when the B-type immediate is negative.
    imm_bucket=0 (-2048) and imm_bucket=1 (-100) both give negative offsets.
    All 6 branch flavours (BEQ/BNE/BLT/BGE/BLTU/BGEU) exercise instr_b=1;
    the negative offset ensures predict_branch_taken_o=1 for each.
    """
    stream = []
    for _ in range(10):
        op = rng.choice(list(range(38, 44)))
        stream.append((op, 0, rng.randint(0, 31), rng.randint(0, 31),
                       rng.choice([0, 1]), 0))   # negative offset → taken
    return stream


def build_ibex_branch_predict_jal_stream(rng):
    """Target ibex_branch_predict.sv: instr_j=1 arm in unique case — always taken.

    assign instr_j = opcode_e'(instr[6:0]) == OPCODE_JAL
    JAL is always predicted taken regardless of sign of offset.
    Test both negative (bucket=0,1) and positive (bucket=3,4) offsets to show
    predict_branch_taken_o=1 in both cases (instr_j dominates instr_b_taken).
    """
    stream = []
    for imm in [0, 1, 2, 3, 4]:
        stream.append((44, rng.randint(1, 31), 0, 0, imm, 0))  # JAL
    for _ in range(6):
        stream.append((44, rng.randint(1, 31), 0, 0, rng.choice([0, 1, 3, 4]), 0))
    return stream


def build_ibex_branch_predict_compressed_jump_stream(rng):
    """Target ibex_branch_predict.sv: instr_cj=1 arm — compressed jump always taken.

    assign instr_cj = (instr[1:0] == 2'b01) & ((instr[15:13] == 3'b101) | (instr[15:13] == 3'b001))

    CORRECTED after review: verified against codec_l9.py's Op enum that of
    the 26-op pool previously sampled (range(45,61)+range(73,83)), only
    C_JAL=73 (quadrant1, funct3=001) and C_J=74 (quadrant1, funct3=101)
    actually satisfy instr_cj — the other 24 ops (C_ADDI..C_ANDI, quadrant-1
    format ops that assert instr_cb instead, quadrant-2 ops, C_EBREAK) do
    not, so the old pool only hit the claimed arm ~2/26 ≈ 8% of the time.
    Narrowed the pool to exactly [73, 74].
    """
    stream = []
    compressed_jump_ops = [73, 74]  # C_JAL, C_J — the only two ops that assert instr_cj
    for _ in range(12):
        op = rng.choice(compressed_jump_ops)
        stream.append((op, rng.randint(0, 31), rng.randint(0, 31), 0,
                       rng.choice([0, 1, 3, 4]), 0))
    return stream


def build_ibex_branch_predict_compressed_branch_stream(rng):
    """Target ibex_branch_predict.sv: instr_cb=1 + imm_cb_type[31] taken path.

    assign instr_cb = (instr[1:0] == 2'b01) & ((instr[15:13] == 3'b110) | (instr[15:13] == 3'b111))
    assign instr_b_taken = ... | (instr_cb & imm_cb_type[31])

    CORRECTED after review: verified against codec_l9.py's Op enum that of
    the 10-op pool previously sampled (range(73,83)), only C_BEQZ=75 and
    C_BNEZ=76 satisfy instr_cb (C_JAL/C_J assert instr_cj instead;
    C_ADDI16SP/C_LWSP/C_SWSP/C_JR/C_JALR/C_EBREAK assert neither) — a 20%
    hit rate for the claimed target. Narrowed the pool to exactly [75, 76].

    imm_cb_type[31] = instr[12] (the sign bit of the CB-format immediate).
    Backward offset (imm_bucket=0 or 1 → negative) sets imm_cb_type[31]=1.
    """
    stream = []
    compressed_branch_ops = [75, 76]  # C_BEQZ, C_BNEZ — the only two ops that assert instr_cb
    for _ in range(12):
        op = rng.choice(compressed_branch_ops)
        stream.append((op, rng.randint(0, 31), rng.randint(0, 31), 0,
                       rng.choice([0, 1]), 0))   # negative offset → taken
    for _ in range(6):
        op = rng.choice(compressed_branch_ops)
        stream.append((op, rng.randint(0, 31), rng.randint(0, 31), 0,
                       rng.choice([3, 4]), 0))   # positive offset → not taken
    return stream


def build_ibex_branch_predict_all_case_arms_stream(rng):
    """Target ibex_branch_predict.sv: all 4 unique case arms in one stream.

    unique case (1'b1)
      instr_j  : branch_imm = imm_j_type;
      instr_b  : branch_imm = imm_b_type;
      instr_cj : branch_imm = imm_cj_type;
      instr_cb : branch_imm = imm_cb_type;
      default  : (branch_imm = imm_b_type, no prediction)

    CORRECTED after review: the line commented `# instr_cj` previously
    sampled range(45,61) (C_ADDI..C_ANDI), which contains ZERO ops that can
    assert instr_cj (0% hit rate) — the two ops that do (C_JAL=73, C_J=74)
    aren't even in that range. The line commented `# instr_cb` sampled
    range(73,83), which includes C_JAL/C_J (these assert instr_cj, not
    instr_cb) plus several ops asserting neither; only 2/10 draws
    (C_BEQZ/C_BNEZ) correctly hit instr_cb. Fixed by swapping in the exact
    op pools verified above: [73, 74] for instr_cj, [75, 76] for instr_cb.
    """
    stream = []
    for _ in range(6):
        stream.append((44, rng.randint(1, 31), 0, 0, rng.choice([1, 3]), 0))          # instr_j
        stream.append((rng.choice(range(38, 44)), 0,
                       rng.randint(0, 31), rng.randint(0, 31), rng.choice([1, 3]), 0)) # instr_b
        stream.append((rng.choice([73, 74]), rng.randint(0, 31),
                       rng.randint(0, 31), 0, rng.choice([1, 3]), 0))                  # instr_cj
        stream.append((rng.choice([75, 76]), rng.randint(0, 31),
                       rng.randint(0, 31), 0, rng.choice([1, 3]), 0))                  # instr_cb
        stream.append((rng.choice([0, 1, 5, 10]), rng.randint(1, 31),
                       rng.randint(1, 31), rng.randint(1, 31), 2, 0))                  # default
    return stream


# ---------------------------------------------------------------------------
# ibex_core.sv (scope "core_a") — top-level glue muxes fed purely by
# instruction content: csr_wdata bus mux, perf_* event wiring, pmp_req_type
# mux, RAW-hazard stall glue. Merged from _draft_streams_core_a.py.
# ---------------------------------------------------------------------------

def build_core_csr_wdata_bus_toggle_stream(rng):
    """Target ibex_core.sv:1055: `assign csr_wdata = alu_operand_a_ex;`

    This is the single top-level wire that feeds cs_registers_i.csr_wdata_i
    for every CSR write in the design. Register-form CSR ops (CSRRW/CSRRS/
    CSRRC, op=27/28/29) select OP_A_REG_A so alu_operand_a_ex carries the
    full 32-bit rs1 value; immediate-form CSR ops (CSRRWI/CSRRSI/CSRRCI,
    op=66/67/68) select IMM_A_Z so alu_operand_a_ex carries only
    zimm_rs1_type — the 5-bit rs1-field uimm zero-extended to 32 bits (top
    27 bits forced to 0). Rapid alternation forces the shared 32-bit bus to
    see wide, unconstrained content on one cycle and a narrow all-upper-zero
    pattern on the next, maximising bit-toggle activity on csr_wdata[31:5].

    MERGED from _draft_streams_core_a.py, with one fix applied during merge:
    the draft's bucket list (0-3, 29-31) included bucket 31 = mtvec
    (verified: L11_CSRS[31]=0x305) while its own docstring claimed these
    buckets "accept arbitrary writes without side-effects that could trap
    execution" — false for mtvec specifically (an unconstrained write there
    relocates the trap base; if any exception fires later in the program,
    the core jumps to a garbage PC). Dropped bucket 31 from the list,
    consistent with the same fix applied to build_ibex_csr_wr_en_stream
    above; buckets 0-3 (mscratch/mepc/mcause/mtval) and 29-30
    (mstatus/mie) have no such trap-redirection side effect.
    """
    stream = []
    buckets = [0, 1, 2, 3, 29, 30]
    for bucket in buckets:
        # Wide, register-sourced write (rs1 = whatever a prior op left there)
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
        # Narrow, immediate-sourced write on the SAME bucket, next cycle
        stream.append((66, rng.randint(1, 31), rng.randint(0, 31), 0, 2, bucket))
        # Wide again via CSRRS (set-bits form, still OP_A_REG_A)
        stream.append((28, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
        # Narrow again via CSRRCI (clear-bits immediate form)
        stream.append((68, rng.randint(1, 31), rng.randint(0, 31), 0, 2, bucket))
    return stream


def build_core_perf_event_interleave_stream(rng):
    """Target ibex_core.sv perf_* wiring: perf_jump, perf_branch, perf_tbranch,
    perf_load, perf_store, perf_mul_wait, perf_div_wait (declared 376-389;
    wired id_stage_i/ex_block_i/load_store_unit_i -> cs_registers_i at
    712-719 and 1154-1167).

    Each of these is a single point-to-point wire, not a case-arm inside any
    one sub-module — it is the ibex_core-level glue connecting "an event
    happened this cycle" in one stage to the HPM-event mux in
    ibex_cs_registers.sv. This stream round-robins through mutually-
    exclusive instruction kinds (jump / taken branch / load / store / mul /
    div) so perf_jump, perf_tbranch, perf_load, perf_store, perf_mul_wait
    and perf_div_wait each pulse high for exactly one retiring instruction
    before a DIFFERENT wire pulses high next, maximising 0->1->0 toggle
    activity on the wires themselves at the ibex_core glue level.

    MERGED from _draft_streams_core_a.py, unchanged.
    """
    stream = []
    kinds = [
        (44, None),   # JAL -> perf_jump
        (38, True),   # BEQ rd==rs (taken) -> perf_branch & perf_tbranch
        (39, True),   # BNE rd!=rs (taken)
        (21, None),   # LW -> perf_load
        (26, None),   # SW -> perf_store
        (30, None),   # MUL -> perf_mul_wait
        (34, None),   # DIV -> perf_div_wait
        (65, None),   # JALR -> perf_jump (register-indirect variant)
    ]
    for _ in range(24):
        op_i, _taken = rng.choice(kinds)
        if op_i in (38, 39):
            rs = rng.randint(1, 31)
            stream.append((op_i, 0, rs, rs, 2, 0))  # rs1==rs2 -> BEQ taken / BNE not-taken
        elif op_i in (44, 65):
            stream.append((op_i, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))
        elif op_i in (21, 26):
            stream.append((op_i, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(0, 4), 0))
        else:
            stream.append((op_i, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    return stream


def build_core_pmp_req_type_toggle_stream(rng):
    """Target ibex_core.sv:1192-1193 (inside `if (PMPEnable) begin : g_pmp`):

      assign pmp_req_type[PMP_D] = data_we_o ? PMP_ACC_WRITE : PMP_ACC_READ;

    This 2-bit enum mux is computed in ibex_core.sv ITSELF, one level above
    ibex_pmp.sv's region-match logic. None of the 8 build_pmp_*_stream
    functions above are written to specifically alternate loads and stores
    on consecutive cycles, so pmp_req_type[PMP_D] mostly sits at one enum
    value for runs of several cycles. This stream deliberately alternates
    single loads and single stores every instruction to force
    pmp_req_type[PMP_D] to flip PMP_ACC_READ<->PMP_ACC_WRITE every cycle.
    All 4 pmpcfg regions are left at their reset value (mode=OFF) so every
    access is permitted by default.

    MERGED from _draft_streams_core_a.py, unchanged.
    """
    stream = []
    load_ops = [19, 20, 21, 22, 23]
    store_ops = [24, 25, 26]
    for _ in range(16):
        stream.append((rng.choice(load_ops), rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), rng.randint(0, 2), 0))
        stream.append((rng.choice(store_ops), rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), rng.randint(0, 2), 0))
    return stream


def build_core_raw_hazard_stall_stream(rng):
    """Target ibex_core.sv:314-316 stall-control wiring: `id_in_ready`,
    `ex_valid` glue between ex_block_i/id_stage_i, and `lsu_resp_valid`
    between load_store_unit_i and id_stage_i.

    ex_valid is driven low by ex_block_i for the duration of a multi-cycle
    MULDIV op; lsu_resp_valid is driven low by the LSU while a load/store is
    outstanding. Both directly gate id_in_ready in ibex_id_stage.sv. This
    stream chains DIV/MUL -> dependent ADD (reusing rd as the next rs1) and
    LW -> dependent ADDI (same pattern) repeatedly, forcing the
    id_in_ready/ex_valid/lsu_resp_valid stall-then-resume glue to toggle on
    every pair of instructions.

    MERGED from _draft_streams_core_a.py, unchanged.
    """
    stream = []
    for _ in range(10):
        rd = rng.randint(1, 31)
        rs = rng.randint(1, 31)
        stream.append((rng.choice([30, 34]), rd, rs, rng.randint(1, 31), 2, 0))  # MUL/DIV
        stream.append((0, rng.randint(1, 31), rd, rng.randint(1, 31), 2, 0))     # ADD uses rd
        rd2 = rng.randint(1, 31)
        stream.append((21, rd2, rs, rng.randint(1, 31), rng.randint(0, 2), 0))   # LW
        stream.append((10, rng.randint(1, 31), rd2, 0, 2, 0))                    # ADDI uses rd2
    return stream


# ---------------------------------------------------------------------------
# ibex_core.sv (scope "core_b") — illegal-instruction wire toggle. Merged
# from _draft_streams_core_b.py. That draft's own investigation (fcov block
# stripped under -DSYNTHESIS=1, RegFileECC=0 hardcoded localparam in
# ibex_top.sv, irq/debug ports tied off in cocotb_ibex_max_upstream.sv) found
# every OTHER candidate in its assigned scope structurally unreachable in
# this build; only this one function survived that audit.
# ---------------------------------------------------------------------------

def build_core_illegal_insn_toggle_stream(rng):
    """Target ibex_core.sv:391,723 -- illegal_insn_id / unused_illegal_insn_id wires.

    ibex_core.sv declares (line 391) `logic illegal_insn_id,
    unused_illegal_insn_id;` and (line 723, unconditional, not inside any
    ifdef) `assign unused_illegal_insn_id = illegal_insn_id;`.
    illegal_insn_id is wired straight through from ibex_id_stage's
    .illegal_insn_o. This is the only elaborated consumer of illegal_insn_id
    in ibex_core.sv (the fcov block that also reads it is stripped under
    -DSYNTHESIS=1, per cpu/Makefile.upstream).

    ILLEGAL_INSN (op=83, codec_l11.py) is a fixed CUSTOM-0 encoding
    (0x0000000B) that falls through the decoder's default case to
    illegal_insn_o=1. Alternating it 1:1 against a cheap legal instruction
    (ADDI, op=10) forces a 0->1->0 transition on illegal_insn_id every 2
    cycles — denser than build_ibex_load_store_misaligned_stream above,
    which only hits op 83 on a ~1-in-4 rng.choice() draw and targets a
    different signal (ibex_load_store_unit.sv's load_err_o/store_err_o).

    MERGED from _draft_streams_core_b.py, unchanged.
    """
    stream = []
    for _ in range(24):
        stream.append((83, 0, 0, 0, 2, 0))  # ILLEGAL_INSN -> illegal_insn_id = 1
        stream.append((10, rng.randint(1, 31), rng.randint(1, 31), 0,
                       rng.randint(0, 4), 0))  # ADDI (legal) -> illegal_insn_id = 0
    return stream


# ---------------------------------------------------------------------------
# ibex_core.sv (scope "core_c") — exception/trap-capture CSR toggle diversity
# + nt_branch_addr PC-walk. Pass 3 (isolated synthesis -> isolated review ->
# fix+merge, same pipeline as passes 1-2). csr_depc/debug_cause confirmed
# structurally unreachable on this build (debug_req_i tied to 0 in the tb,
# DCSR/DPC/trigger-register writes all gated behind debug_mode_i already
# being set -- a closed loop with no entry point) -- no stream synthesized
# for those two, reported as a finding instead.
# ---------------------------------------------------------------------------

def _emit_diverse_value(stream, rng, dst):
    """Load two distinct registers via ADDI, SLL one by the other, XOR against
    a third ADDI-loaded register, to scramble the 5 fixed imm_bucket literals
    ({-2048,-100,0,100,2047}) into a much richer 32-bit pattern in `dst` than
    any single literal alone -- used by the CSR-toggle streams below, whose
    target registers pass csr_wdata_int through mostly/fully unmasked, so
    value diversity (not reachability) was the limiting factor.
    """
    a = rng.randint(1, 31)
    b = rng.randint(1, 31)
    c = rng.randint(1, 31)
    while b == a:
        b = rng.randint(1, 31)
    stream.append((10, a, 0, 0, rng.randint(0, 4), 0))
    stream.append((10, b, 0, 0, rng.randint(0, 4), 0))
    stream.append((2,  a, a, b, 0, 0))   # SLL a, a, b
    stream.append((10, c, 0, 0, rng.randint(0, 4), 0))
    stream.append((5,  dst, a, c, 0, 0))  # XOR dst, a, c


def build_core_csr_mepc_toggle_stream(rng):
    """Target ibex_cs_registers.sv:227 (mepc_q, mepc_d), :609-610 (default
    mepc_d = {csr_wdata_int[31:1], 1'b0}), :669 (CSR_MEPC: mepc_en=1'b1),
    :920-932 (mepc_q register instance), :881 (csr_mepc_o = mepc_q). Also
    covers ibex_core.sv:966 crash_dump_o.exception_pc (a pure alias of mepc).

    CSR_MEPC (0x341) is not in ibex_cs_registers.sv's dbg_csr set, so
    illegal_csr_dbg never blocks this write from ordinary M-mode code.
    csr_bucket=1 -> L11_CSRS[1] == 0x341 (mepc), verified against the CSR
    pool's build order (L7_SAFE_CSRS[1] = mepc; L11_CSRS = ... + L7_SAFE_CSRS
    + ... unmodified through L9/L10/L11).

    Caveat: mepc_d forces bit0 to 0 on every write, so mepc's bit0 toggle
    coverage is structurally capped regardless of stimulus; bits[31:1] get
    full diversity from _emit_diverse_value.
    """
    CSR_MEPC_BUCKET = 1
    stream = []
    for _ in range(24):
        dst = rng.randint(1, 31)
        _emit_diverse_value(stream, rng, dst)
        stream.append((27, 0, dst, 0, 0, CSR_MEPC_BUCKET))  # CSRRW x0, mepc, dst
    return stream


def build_core_csr_mtvec_toggle_stream(rng):
    """Target ibex_cs_registers.sv:233 (mtvec_q, mtvec_d), :617-621 (only
    bits[31:8] are software-controlled -- bits[7:2] hardzero, bits[1:0]
    fixed to 2'b01 -- on both the reset-init and explicit-write paths),
    :678 (CSR_MTVEC: mtvec_en=1'b1), :994-1006 (mtvec_q instance), :883
    (csr_mtvec_o=mtvec_q). Not in dbg_csr.

    csr_bucket=31 -> L11_CSRS[31] == 0x305 (mtvec): L9_CSRS appends
    [mstatus, mie, mtvec] after L7_SAFE_CSRS's 29 entries (indices 0-28),
    landing mstatus=29, mie=30, mtvec=31; L10/L11 only append further CSRs
    after this point, so the index is stable.

    Caveat: bits[7:0] are pinned to 0x01 by RTL on every write (same as
    reset), so only bits[31:8] (24 bits) carry real coverage value here.
    """
    CSR_MTVEC_BUCKET = 31
    stream = []
    for _ in range(24):
        dst = rng.randint(1, 31)
        _emit_diverse_value(stream, rng, dst)
        stream.append((27, 0, dst, 0, 0, CSR_MTVEC_BUCKET))  # CSRRW x0, mtvec, dst
    return stream


def build_core_crash_dump_exception_addr_toggle_stream(rng):
    """Target ibex_cs_registers.sv:231 (mtval_q, mtval_d), :615-616 (default
    mtval_en=1'b0; mtval_d=csr_wdata_int, fully UNMASKED -- unlike mepc/
    mtvec above), :675 (CSR_MTVAL: mtval_en=1'b1), :980-992 (mtval_q
    instance), :884 (csr_mtval_o=mtval_q). ibex_core.sv:967
    crash_dump_o.exception_addr = crash_dump_mtval, wired at ibex_core.sv:
    1110 (.csr_mtval_o(crash_dump_mtval)) -- confirmed a genuine mtval alias.

    csr_bucket=3 -> L11_CSRS[3] == 0x343 (mtval): L7_SAFE_CSRS[0..3] =
    mscratch, mepc, mcause, mtval.
    """
    CSR_MTVAL_BUCKET = 3
    stream = []
    for _ in range(16):
        dst = rng.randint(1, 31)
        _emit_diverse_value(stream, rng, dst)
        stream.append((27, 0, dst, 0, 0, CSR_MTVAL_BUCKET))  # CSRRW x0, mtval, dst
    return stream


def build_core_nt_branch_addr_toggle_stream(rng):
    """Target ibex_core.sv:222 (nt_branch_addr), driven at ibex_id_stage.sv:
    757-758: nt_branch_addr_o = pc_id_i + (instr_is_compressed_i ? 2 : 4),
    UNCONDITIONALLY for every instruction (not gated on branch/jump opcode),
    when BranchPredictor=1 -- confirmed forced to 1 in this project's build
    (cpu/cocotb_ibex_max_upstream.sv:52), else tied to 0.

    Since nt_branch_addr is combinationally pc_id_i + a small constant, its
    bit pattern is a function of the live PC. This stream chains many
    always-taken control-flow ops with randomized imm_bucket offsets so the
    PC executes a long biased random walk, spreading pc_id_i's (and hence
    nt_branch_addr's) bit pattern across many positions: JAL (op=44,
    uncompressed, +4 addend case), C_J (op=74, compressed, +2 addend case),
    BEQ (op=38) with rs1==rs2 (always taken, any register value), and
    C_BEQZ (op=75) against x8.

    C_BEQZ note: RVC's 3-bit prime-register field (rs1p = rs1 & 7) can only
    address x8-x15, NOT x0 -- passing rs1=0 selects x8, not x0. x8 is
    zeroed once at stream start via ADDI so the C_BEQZ branch is a real,
    guaranteed-taken branch rather than testing an arbitrary register.
    """
    JAL, C_J, BEQ, C_BEQZ, ADDI = 44, 74, 38, 75, 10
    stream = [(ADDI, 8, 0, 0, 2, 0)]  # x8 = 0, so C_BEQZ (x8) below is real
    for _ in range(40):
        choice = rng.randint(0, 3)
        bucket = rng.randint(0, 4)
        if choice == 0:
            stream.append((JAL, rng.randint(1, 31), 0, 0, bucket, 0))
        elif choice == 1:
            stream.append((C_J, 0, 0, 0, bucket, 0))
        elif choice == 2:
            r = rng.randint(1, 31)
            stream.append((BEQ, 0, r, r, bucket, 0))
        else:
            stream.append((C_BEQZ, 0, 0, 0, bucket, 0))  # tests x8 (==0)
    return stream


# ---------------------------------------------------------------------------
# ibex_decoder.sv — RV32B decode case-arms not exercised by build_alu_rv32b_stream
# (which only covers ROL/ROR/RORI/BSET/BCLR/BINV/BEXT/SH1ADD/SH2ADD/SH3ADD).
# Merged from _draft_streams_decoder.py.
# ---------------------------------------------------------------------------

def build_decoder_op_imm_bitcount_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP_IMM, funct3=001, instr[31:27]=5'b01100
    nested `unique case(instr[26:20])` bit-count/sign-extend group (lines
    381-397): clz(102)/ctz(103)/cpop(104)/sext.b(105)/sext.h(106).

    None of these 5 case items are hit by any other function (only the
    sibling instr[31:27] arms bclri/bseti/binvi/bexti/rol/ror are). This
    stream cycles op=102..106 across varied rs1/rd so the nested-case
    comparator on instr[26:20] toggles through all 5 legal values.

    MERGED from _draft_streams_decoder.py, unchanged.
    """
    CLZ, CTZ, CPOP, SEXT_B, SEXT_H = 102, 103, 104, 105, 106
    ADDI = 10
    stream = []
    src_regs = [rng.randint(1, 8) for _ in range(3)]
    for i, r in enumerate(src_regs):
        stream.append((ADDI, r, 0, 0, i % 5, 0))
    for op in (CLZ, CTZ, CPOP, SEXT_B, SEXT_H):
        for r in src_regs:
            stream.append((op, rng.randint(1, 31), r, 0, rng.randint(0, 4), 0))
    return stream


def build_decoder_crc32_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP_IMM, funct3=001, instr[31:27]=5'b01100,
    instr[26:20] crc32/crc32c sub-group (lines 388-395) — the narrower-gated
    `default` arm of the bit-count nested case above (legal only for
    RV32BOTEarlGrey/RV32BFull, not RV32BBalanced). Also drives
    ibex_alu.sv's ALU_CRC32_* operators with alu_multicycle_o=1.

    MERGED from _draft_streams_decoder.py, unchanged.
    """
    CRC32_B, CRC32_H, CRC32_W = 137, 138, 139
    CRC32C_B, CRC32C_H, CRC32C_W = 140, 141, 142
    stream = []
    for op in (CRC32_B, CRC32_H, CRC32_W, CRC32C_B, CRC32C_H, CRC32C_W):
        for rs1 in (0, 1, 15, 31):
            stream.append((op, rng.randint(1, 31), rs1, 0, rng.randint(0, 4), 0))
    return stream


def build_decoder_op_imm_shamt_family_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP_IMM shamt-immediate case arms never
    exercised elsewhere: sloi(131)/bclri(112)/bseti(113)/binvi(114)/
    shfli(135)/sroi(132)/bexti(115)/grevi(133)/gorci(134)/unshfli(136).
    Sweeps shamt across all 5 SHAMT_BUCKET_VALUES {0,8,16,24,31} for every
    op so the imm[24:20] shamt field toggles fully for each case arm.

    MERGED from _draft_streams_decoder.py, unchanged.
    """
    BCLRI, BSETI, BINVI, BEXTI = 112, 113, 114, 115
    SLOI, SROI, GREVI, GORCI, SHFLI, UNSHFLI = 131, 132, 133, 134, 135, 136
    stream = []
    for op in (SLOI, BCLRI, BSETI, BINVI, SHFLI, SROI, BEXTI, GREVI, GORCI, UNSHFLI):
        for imm_bucket in range(5):
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0, imm_bucket, 0))
    return stream


def build_decoder_zbp_rtype_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP, `unique case ({instr[31:25],
    instr[14:12]})`, zbp (legacy draft bitmanip permutation) block (lines
    495-504): slo(116)/sro(117)/grev(118)/gorc(119)/shfl(120)/unshfl(121)/
    xperm.n(122)/xperm.b(123)/xperm.h(124). 9 case items entirely unvisited
    by build_alu_rv32b_stream (ops 116-124 never touched there).

    MERGED from _draft_streams_decoder.py, unchanged.
    """
    SLO, SRO = 116, 117
    GREV, GORC = 118, 119
    SHFL, UNSHFL = 120, 121
    XPERM_N, XPERM_B, XPERM_H = 122, 123, 124
    stream = []
    for op in (SLO, SRO, GREV, GORC, SHFL, UNSHFL, XPERM_N, XPERM_B, XPERM_H):
        for _ in range(3):
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    for op in (SLO, GREV, SHFL, XPERM_N, XPERM_H):
        stream.append((op, 31, 0, 31, 2, 0))
        stream.append((op, 1, 31, 0, 2, 0))
    return stream


def build_decoder_zbc_clmul_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP, zbc carry-less-multiply block
    (lines 505-510): clmul(125)/clmulr(126)/clmulh(127). Shares funct7
    (0000101) with MIN/MAX/MINU/MAXU but uses funct3 001/010/011 instead —
    a comparator slice no existing function drives (MIN/MAX family only
    appears with funct3 100/101/110/111).

    MERGED from _draft_streams_decoder.py, unchanged.
    """
    CLMUL, CLMULR, CLMULH = 125, 126, 127
    stream = []
    for op in (CLMUL, CLMULR, CLMULH):
        for _ in range(4):
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    return stream


def build_decoder_zbe_zbf_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP, zbe (bit-compress/decompress,
    RV32BFull-only) and zbf (bit-field-place) blocks: bcompress(128)/
    bdecompress(129)/bfp(130).

    MERGED from _draft_streams_decoder.py, unchanged.
    """
    BCOMPRESS, BDECOMPRESS = 128, 129
    BFP = 130
    stream = []
    for op in (BCOMPRESS, BDECOMPRESS):
        for _ in range(4):
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    for _ in range(6):
        stream.append((BFP, rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), 2, 0))
    return stream


def build_decoder_zbb_logic_pack_minmax_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP, zbb logic/pack/min-max case items
    (lines 476-487) not already covered by build_alu_compare_sign_mismatch_stream:
    andn(90)/orn(91)/xnor(92), min(95)/max(96)/minu(97)/maxu(98) (full sweep,
    not just the narrow mixed-sign slice), pack(99)/packh(100)/packu(101).

    MERGED from _draft_streams_decoder.py, unchanged.
    """
    ANDN, ORN, XNOR = 90, 91, 92
    MIN_OP, MAX_OP, MINU_OP, MAXU_OP = 95, 96, 97, 98
    PACK, PACKH, PACKU = 99, 100, 101
    ADDI = 10
    stream = []
    for op in (ANDN, ORN, XNOR, PACK, PACKH, PACKU):
        for _ in range(4):
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    neg_r = rng.randint(1, 15)
    pos_r = rng.randint(16, 31)
    stream.append((ADDI, neg_r, 0, 0, 0, 0))  # -2048 -> bit[31]=1
    stream.append((ADDI, pos_r, 0, 0, 4, 0))  # +2047 -> bit[31]=0
    for op in (MAX_OP, MAXU_OP, MIN_OP, MINU_OP):
        stream.append((op, rng.randint(1, 31), neg_r, pos_r, 2, 0))
        stream.append((op, rng.randint(1, 31), pos_r, neg_r, 2, 0))
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), 2, 0))
    return stream


def build_decoder_illegal_default_reset_stream(rng):
    """Target ibex_decoder.sv: top-level opcode case `default: illegal_insn
    = 1'b1;` (line 643-645) and the reset-on-illegal gating block (lines
    658-666) that forces rf_we/data_req_o/data_we_o/jump_in_dec_o/
    jump_set_o/branch_in_dec_o/csr_access_o to 0 whenever illegal_insn=1.

    This stream interleaves op 83 (ILLEGAL_INSN, the codec's only way to
    reach this default arm) with instructions that set each of the seven
    gated signals immediately before/after, so the gating mux for each
    signal toggles 0(legal)->1(illegal, forced 0)->0(legal) across
    consecutive cycles.

    MERGED from _draft_streams_decoder.py, unchanged.
    """
    ILLEGAL_INSN = 83
    ADD, JAL, BEQ = 0, 44, 38
    SW, LW = 26, 21
    CSRRW = 27
    stream = []
    setters = [
        (ADD, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0),
        (JAL, rng.randint(1, 31), 0, 0, 2, 0),
        (SW, 0, rng.randint(1, 31), rng.randint(1, 31), 2, 0),
        (LW, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0),
        (BEQ, 0, rng.randint(1, 31), rng.randint(1, 31), 2, 0),
        (CSRRW, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0),
    ]
    for setter in setters:
        stream.append(setter)
        stream.append((ILLEGAL_INSN, 0, 0, 0, 0, 0))
        stream.append(setter)
    for _ in range(3):
        stream.append((ILLEGAL_INSN, 0, 0, 0, 0, 0))
        stream.append((ADD, rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), 2, 0))
    return stream


# ---------------------------------------------------------------------------
# ibex_decoder.sv — bt_a_mux_sel_o diversity. Pass 3 (isolated synthesis ->
# isolated review -> fix+merge). 9 of the 11 originally-targeted decoder
# gaps confirmed structurally unreachable on this codec/build (hardwired-
# zero RTL bits in imm_u_type_o/imm_b_type_o/imm_j_type_o, RV32E forced 0,
# csr_illegal's funct3 pattern never emitted, use_rs3 needing an rs3 field
# the 6-tuple action space doesn't have -- codec_l11.py's own docstring
# already flags this, dret_insn_o needing an opcode never in the 143-op
# table, illegal_c_insn_i's trigger conditions actively engineered out of
# codec_rvc.py) -- reported as findings, no stream synthesized for those.
# ---------------------------------------------------------------------------

def build_decoder_bt_a_mux_sel_reg_a_diversity_stream(rng):
    """Target ibex_decoder.sv bt_a_mux_sel_o (op_a_sel_e 2-bit enum,
    ibex_pkg.sv:246-251: OP_A_REG_A=00, OP_A_FWD=01, OP_A_CURRPC=10,
    OP_A_IMM=11). Traced every assignment in the ALU-control always_comb
    block (ibex_decoder.sv:673-1189): default = OP_A_CURRPC (line 681,
    LOAD/STORE/OP_IMM/OP/LUI/AUIPC/SYSTEM never touch it); explicit
    reassignment only in OPCODE_JAL (line 700, CURRPC), OPCODE_JALR (line
    722, REG_A), OPCODE_BRANCH (line 755, CURRPC), and FENCE.I (line 1153,
    CURRPC) -- all gated by BranchTargetALU, confirmed 1 in
    cpu/cocotb_ibex_max_upstream.sv:47.

    OP_A_FWD/OP_A_IMM never appear anywhere -> bit[0] of bt_a_mux_sel_o is
    permanently 0, unreachable with any stimulus. This stream targets
    bit[1] (CURRPC vs REG_A) by densely alternating JALR (the only op that
    ever sets REG_A) against JAL/BEQ/FENCE.I (all CURRPC), in case JALR was
    under-exercised relative to its siblings by the rest of the CRT.
    """
    JAL, JALR, BEQ, FENCE_I = 44, 65, 38, 72
    stream = []
    for _ in range(10):
        stream.append((JALR, rng.randint(0, 31), rng.randint(1, 31), 0,
                        rng.randint(0, 4), 0))
        choice = rng.choice([JAL, BEQ, FENCE_I])
        if choice == BEQ:
            stream.append((BEQ, 0, rng.randint(0, 31), rng.randint(0, 31),
                            rng.randint(0, 4), 0))
        elif choice == JAL:
            stream.append((JAL, rng.randint(0, 31), 0, 0,
                            rng.randint(0, 4), 0))
        else:
            stream.append((FENCE_I, 0, 0, 0, 0, 0))
    return stream


# ---------------------------------------------------------------------------
# ibex_compressed_decoder.sv — RVC decode case-arms. Merged from
# _draft_streams_compressed_decoder.py.
# ---------------------------------------------------------------------------

def build_compressed_c1_alu_shift_ca_mux_stream(rng):
    """Target ibex_compressed_decoder.sv: C1 quadrant, funct3=100 nested case
    (lines 366-486): c.srli(58)/c.srai(59)/c.andi(60) via instr_i[11:10],
    and c.sub(57)/c.xor(56)/c.or(55)/c.and(54) via the CA-format
    {instr_i[12],instr_i[6:5]} sub-case. (c.subw/c.addw are RV64-only,
    illegal unconditionally; c.mul/c.zext.*/c.sext.*/c.not are Zcb-gated —
    none of these six leaves has a codec op index, so they're structurally
    unreachable here and are skipped rather than faked.)

    MERGED from _draft_streams_compressed_decoder.py, unchanged.
    """
    stream = []
    ops = [58, 59, 60, 57, 56, 55, 54]  # C_SRLI, C_SRAI, C_ANDI, C_SUB, C_XOR, C_OR, C_AND
    for ib in range(5):
        for op in ops:
            rd = rng.randint(8, 15)
            rs1 = rng.randint(8, 15)
            rs2 = rng.randint(8, 15)
            stream.append((op, rd, rs1, rs2, ib, 0))
    return stream


def build_compressed_c2_reg_ctrl_mux_stream(rng):
    """Target ibex_compressed_decoder.sv: C2 quadrant, funct3=100 register-
    control if/else tree (lines 523-549): c.mv(49)/c.jr(80)/c.add(50)/
    c.ebreak(82)/c.jalr(81), gated by instr_i[12] and instr_i[6:2]!=0 (plus
    a nested instr_i[11:7]==0 check inside the bit12=1/rs2==0 branch).
    Round-robin forces every leaf every cycle.

    The rd==0 illegal check on c.jr (line 532) is unreachable through the
    codec: the C_JR encoder floors rs1 to >=1 unconditionally, so it is
    skipped.

    MERGED from _draft_streams_compressed_decoder.py, unchanged.
    """
    stream = []
    ops = [49, 80, 50, 82, 81]  # C_MV, C_JR, C_ADD, C_EBREAK, C_JALR
    for _ in range(10):
        for op in ops:
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), rng.randint(0, 4), 0))
    return stream


def build_compressed_lui_addi16sp_special_case_stream(rng):
    """Target ibex_compressed_decoder.sv: C1 quadrant funct3=011 rd-field
    override mux (lines 352-364) — a single rd-field comparator re-routes
    the SAME quadrant/funct3 encoding from a U-type LUI construction
    (default) to a completely different I-type stack-pointer-ADDI
    construction when instr_i[11:7]==5'h02.

    C_LUI(47) always encodes rd in {3..31} (avoids rd in {0,2}), so on its
    own it only ever exercises the default LUI path. C_ADDI16SP(77)
    hardcodes rd=5'h02 unconditionally, so it only ever exercises the
    override path. Alternating the two ops forces this mux to flip on
    every pair of actions.

    The illegal (nzimm==0) leaf is unreachable via either op (both
    encoders are constructed to never produce nzimm==0) — skipped rather
    than faked.

    MERGED from _draft_streams_compressed_decoder.py, unchanged.
    """
    stream = []
    for ib in range(5):
        stream.append((47, rng.randint(3, 31), 0, 0, ib, 0))  # C_LUI -> default LUI path
        stream.append((77, 2, 0, 0, ib, 0))                   # C_ADDI16SP -> override path
    return stream


def build_compressed_cl_cs_immediate_scramble_stream(rng):
    """Target ibex_compressed_decoder.sv: C0 quadrant c.lw(52)/c.sw(53)
    immediate bit-scramble (lines 235-246) — the 7-bit byte offset is NOT
    contiguous in the RVC encoding (imm[5:3] at instr_i[12:10], imm[2] at
    instr_i[6], imm[6] at instr_i[5]), a genuine bit permutation. Cycling
    rd'/rs1'/rs2' through all 8 prime registers (x8-x15) exercises the
    3-bit register-select fields this case shares with the CA/CB group.

    MERGED from _draft_streams_compressed_decoder.py, unchanged.
    """
    stream = []
    for ib in range(5):
        for base in range(8):
            rd = 8 + base
            rs1 = 8 + ((base + 3) % 8)
            rs2 = 8 + ((base + 5) % 8)
            stream.append((52, rd, rs1, 0, ib, 0))    # C_LW
            stream.append((53, 0, rs1, rs2, ib, 0))   # C_SW
    return stream


def build_compressed_ci_addi_li_signext_stream(rng):
    """Target ibex_compressed_decoder.sv: C1 quadrant funct3=000 (c.addi,
    op=45) and funct3=010 (c.li, op=46) CI-format sign-bit fanout (lines
    328-350) — both replicate the single CI sign bit (instr_i[12]) across 7
    output bits via `{6{instr_i[12]}}`. rd=0 with imm=0 on c.addi is
    architecturally c.nop — still legal (no illegal_instr_o anywhere in
    this arm), so this stream deliberately includes that hint encoding
    rather than avoiding it.

    MERGED from _draft_streams_compressed_decoder.py, unchanged.
    """
    stream = []
    for ib in range(5):
        stream.append((45, 0, 0, 0, ib, 0))                     # C_ADDI, rd=0 (c.nop hint when ib=2 -> imm=0)
        stream.append((45, rng.randint(1, 31), 0, 0, ib, 0))    # C_ADDI, rd!=0
        stream.append((46, rng.randint(0, 31), 0, 0, ib, 0))    # C_LI
    return stream


def build_compressed_ciw_addi4spn_scramble_stream(rng):
    """Target ibex_compressed_decoder.sv: C0 quadrant funct3=000
    c.addi4spn(51) (lines 226-233) — the file's most scrambled immediate
    field, stored as bit order [5|4|9|8|7|6|2|3] across instr_i[10:5]. rd'
    cycles through all 8 prime registers (instr_i[4:2]); base reg is
    hardwired to x2/sp by the RTL itself so no rs1 sweep applies here.

    The illegal (nzuimm==0) leaf is unreachable via the codec (its encoder
    asserts nzuimm != 0 before encoding) — skipped rather than faked.

    MERGED from _draft_streams_compressed_decoder.py, unchanged.
    """
    stream = []
    for ib in range(5):
        for rd in range(8, 16):
            stream.append((51, rd, 0, 0, ib, 0))  # C_ADDI4SPN
    return stream


# ---------------------------------------------------------------------------
# ibex_icache.sv — content/CSR-driven config + skid-buffer + fill-density
# mechanics (the address-driven core cache mechanics -- way selection,
# tag-RAM hit/miss, ECC checking -- are unreachable from this codec, which
# has no label/jump-target resolution: imm_bucket only selects one of 5
# fixed offsets relative to the current PC). Merged from
# _draft_streams_icache.py.
#
# icache_enable_i bit position (cpuctrl bit0) confirmed directly against
# ibex_cs_registers.sv's cpu_ctrl_sts_part_t (see the ibex_dummy_instr
# section above for the full struct) -- bit0 IS icache_enable, so this
# draft's claim (unlike pass-1's dummy_instr claim of the same bit) is
# correct as written; no fix needed for the bit position.
# ---------------------------------------------------------------------------

CPUCTRL_BUCKET = 68  # L11_CSRS index of 0x7C0 (cpuctrl); bit0 = icache_enable
FENCE_I = 72
CSRRSI = 67
CSRRCI = 68


def build_ibex_icache_enable_toggle_stream(rng):
    """Target ibex_icache.sv: icache_enable_i central mux points
    (lookup_actual_ic0, fill_cache_new, fill_spec_req, instr_req,
    fill_cache_d gating). CSRRSI x0,cpuctrl,uimm=1 sets bit0=1 (enable);
    CSRRCI x0,cpuctrl,uimm=1 clears bit0 (disable) — both literal-uimm ops,
    so the write value doesn't depend on any register content. Interleaves
    short instruction bursts so the enable/disable toggle lands both
    between fetches and, on short bursts, plausibly mid-fill.

    MERGED from _draft_streams_icache.py, unchanged.
    """
    stream = []
    for round_i in range(8):
        stream.append((CSRRSI, 0, 1, 0, 2, CPUCTRL_BUCKET))
        for _ in range(3):
            op = rng.choice([19, 20, 21, 22, 23])  # LB/LH/LW/LBU/LHU
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0,
                           rng.randint(0, 4), 0))
        stream.append((CSRRCI, 0, 1, 0, 2, CPUCTRL_BUCKET))
        for _ in range(2):
            op = rng.choice([0, 1, 8, 9])  # ADD/SUB/OR/AND — plain fetches while disabled
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    stream.append((CSRRSI, 0, 1, 0, 2, CPUCTRL_BUCKET))  # end enabled
    return stream


def build_ibex_icache_fencei_reinvalidate_burst_stream(rng):
    """Target ibex_icache.sv: inval_state_e FSM, INVAL_CACHE restart arm —
    a later FENCE.I landing while inval_state_q is still INVAL_CACHE from a
    previous one restarts the walk with a new scramble key instead of
    completing it. FENCE.I (op=72) fires icache_inval_o for one cycle on
    its first retire cycle. Firing FENCE.I back-to-back (short ALU gaps, no
    loads) keeps successive icache_inval_i pulses close together, making
    the restart arm likely; a final isolated FENCE.I with a long ALU tail
    lets one invalidation run to completion (IDLE arm) instead.

    MERGED from _draft_streams_icache.py, unchanged.
    """
    stream = []
    for _ in range(10):
        stream.append((FENCE_I, 0, 0, 0, 0, 0))
        for _ in range(2):
            op = rng.choice([0, 1, 5, 8, 9])  # single-cycle ALU, no memory stalls
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    stream.append((FENCE_I, 0, 0, 0, 0, 0))
    for _ in range(40):
        op = rng.choice([0, 1, 5, 8, 9, 10])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), rng.randint(0, 4), 0))
    return stream


def build_ibex_icache_enable_during_invalidate_stream(rng):
    """Target ibex_icache.sv: icache_enable_i x icache_inval_i cross terms
    (fill_cache_d gating, inval_block_cache forcing lookup_actual_ic0/
    fill_cache_new to 0 while an invalidation is in flight). Interleaves
    cpuctrl bit0 toggles with FENCE.I so the two independently-sourced
    config signals overlap in every combination: enable+no-inval,
    enable+inval-in-flight, disable+inval-in-flight, disable+no-inval.

    MERGED from _draft_streams_icache.py, unchanged.
    """
    stream = []
    pattern = [
        (CSRRSI, 1),   # enable
        (FENCE_I, None),  # start invalidation while enabled
        (CSRRCI, 1),   # disable while invalidation is (likely) still in flight
        (FENCE_I, None),  # re-request invalidation while disabled
        (CSRRSI, 1),   # re-enable while invalidation is (likely) still in flight
    ]
    for _ in range(6):
        for op, uimm in pattern:
            if op == FENCE_I:
                stream.append((FENCE_I, 0, 0, 0, 0, 0))
            else:
                stream.append((op, 0, uimm, 0, 2, CPUCTRL_BUCKET))
            alu_op = rng.choice([0, 1, 8, 9])
            stream.append((alu_op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    stream.append((CSRRSI, 0, 1, 0, 2, CPUCTRL_BUCKET))  # leave enabled
    return stream


def build_ibex_icache_compressed_alignment_skid_stream(rng):
    """Target ibex_icache.sv: skid-buffer / output-address-parity FSM.
    output_addr_q[1] (halfword parity) is advanced purely by whether each
    successive retired instruction was 2B (compressed) or 4B — a property
    entirely of which ops the codec chose to emit, independent of any fetch
    address. An ODD run of compressed ops flips parity to 1, so the next
    uncompressed (4B) op is fetched unaligned and must be split across two
    16-bit output beats via the skid buffer. This stream forces that by
    emitting runs of 1, 3, 5 compressed ops (always odd) followed by two
    uncompressed ops, cycling the skid buffer engage/steady-state/disengage
    transitions on every group.

    MERGED from _draft_streams_icache.py, unchanged.
    """
    stream = []
    compressed_ops = [45, 46, 48, 54, 55, 56, 57, 60]  # C_ADDI/C_LI/C_SLLI/C_AND/C_OR/C_XOR/C_SUB/C_ANDI
    uncompressed_ops = [0, 1, 5, 8, 9, 10, 13, 14, 15]  # ADD/SUB/XOR/OR/AND/ADDI/XORI/ORI/ANDI
    for odd_run in [1, 3, 5, 1, 3]:
        for _ in range(odd_run):
            op = rng.choice(compressed_ops)
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))
        op = rng.choice(uncompressed_ops)
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), 2, 0))
        op = rng.choice(uncompressed_ops)
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), 2, 0))
    return stream


def build_ibex_icache_branch_density_stale_fill_stream(rng):
    """Target ibex_icache.sv: fill_stale_d + branch-clears-skid-buffer paths.
    The codec cannot choose a branch's TARGET (imm_bucket is one of 5 fixed
    small offsets, no label resolution), but it fully controls branch/jump
    DENSITY relative to memory ops that start fills. This stream maximises
    that density: loads to start fills (multi-beat, so fill_busy_q stays
    set for several cycles) immediately followed by a branch/jump,
    repeated, so branch_i is statistically likely to assert while the
    preceding load's fill buffer is still busy.

    MERGED from _draft_streams_icache.py, unchanged.
    """
    stream = []
    load_ops = [19, 20, 21, 22, 23]  # LB/LH/LW/LBU/LHU
    branch_ops = [38, 39, 40, 41, 42, 43]  # BEQ/BNE/BLT/BGE/BLTU/BGEU
    jump_ops = [44, 65]  # JAL/JALR
    compressed_branch_jump = [73, 74, 75, 76, 80, 81]  # C_JAL/C_J/C_BEQZ/C_BNEZ/C_JR/C_JALR
    for _ in range(16):
        op = rng.choice(load_ops)
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0,
                       rng.randint(0, 4), 0))
        choice_kind = rng.choice(["branch", "jump", "cjump"])
        if choice_kind == "branch":
            op = rng.choice(branch_ops)
            stream.append((op, 0, rng.randint(0, 31), rng.randint(0, 31),
                           rng.randint(0, 4), 0))
        elif choice_kind == "jump":
            op = rng.choice(jump_ops)
            stream.append((op, rng.randint(1, 31), rng.randint(0, 31), 0,
                           rng.randint(0, 4), 0))
        else:
            op = rng.choice(compressed_branch_jump)
            stream.append((op, rng.randint(0, 31), rng.randint(0, 31), 0,
                           rng.randint(0, 4), 0))
    return stream


# ---------------------------------------------------------------------------
# Convenience: all streams as a list for external callers
# ---------------------------------------------------------------------------

ALL_STREAM_BUILDERS = [
    # ibex_pmp (8)
    build_pmp_mode_sweep_stream,
    build_pmp_permission_check_stream,
    build_pmp_lock_stream,
    build_pmp_tor_boundary_stream,
    build_pmp_napot_stream,
    build_pmp_csrr_variants_stream,
    build_pmp_random_walk_stream,
    build_pmp_all_stream,
    # ibex_cs_registers (3)
    build_csr_full_sweep_stream,
    build_exception_save_restore_stream,
    build_mstatus_bits_stream,
    # ibex_csr (2 -- build_csr_shadow_stream deleted, RTL unreachable, see header)
    build_ibex_csr_wr_en_stream,
    build_ibex_csr_reset_path_stream,
    # ibex_counter (5)
    build_counter_write_stream,
    build_counter_write_high_stream,
    build_counter_inc_stream,
    build_counter_inhibit_stream,
    build_counter_hpmcounter_stream,
    # ibex_dummy_instr (5)
    build_dummy_instr_enable_stream,
    build_dummy_instr_seed_stream,
    build_ibex_dummy_instr_lfsr_arm_sweep_stream,
    build_ibex_dummy_instr_mask_stream,
    build_ibex_dummy_instr_combined_stream,
    # ibex_alu (2)
    build_alu_compare_sign_mismatch_stream,
    build_alu_rv32b_stream,
    # ibex_multdiv_fast (4)
    build_multdiv_signed_fsm_stream,
    build_multdiv_mulh_stream,
    build_ibex_multdiv_fast_div_by_zero_stream,
    build_ibex_multdiv_fast_interleaved_stream,
    # ibex_load_store_unit (4)
    build_load_store_size_stream,
    build_ibex_load_store_sign_ext_stream,
    build_ibex_load_store_misaligned_stream,
    build_ibex_load_store_be_sweep_stream,
    # ibex_branch_predict (7)
    build_branch_predict_stream,
    build_ibex_branch_predict_backward_branch_stream,
    build_ibex_branch_predict_jal_stream,
    build_ibex_branch_predict_compressed_jump_stream,
    build_ibex_branch_predict_compressed_branch_stream,
    build_ibex_branch_predict_all_case_arms_stream,
    # ibex_core, scope core_a (4) -- merged from _draft_streams_core_a.py
    build_core_csr_wdata_bus_toggle_stream,
    build_core_perf_event_interleave_stream,
    build_core_pmp_req_type_toggle_stream,
    build_core_raw_hazard_stall_stream,
    # ibex_core, scope core_b (1) -- merged from _draft_streams_core_b.py
    build_core_illegal_insn_toggle_stream,
    # ibex_core, scope core_c (4) -- Pass 3, exception/trap-capture CSRs +
    # nt_branch_addr PC-walk
    build_core_csr_mepc_toggle_stream,
    build_core_csr_mtvec_toggle_stream,
    build_core_crash_dump_exception_addr_toggle_stream,
    build_core_nt_branch_addr_toggle_stream,
    # ibex_decoder (8) -- merged from _draft_streams_decoder.py
    build_decoder_op_imm_bitcount_stream,
    build_decoder_crc32_stream,
    build_decoder_op_imm_shamt_family_stream,
    build_decoder_zbp_rtype_stream,
    build_decoder_zbc_clmul_stream,
    build_decoder_zbe_zbf_stream,
    build_decoder_zbb_logic_pack_minmax_stream,
    build_decoder_illegal_default_reset_stream,
    # ibex_decoder (1) -- Pass 3, bt_a_mux_sel_o diversity
    build_decoder_bt_a_mux_sel_reg_a_diversity_stream,
    # ibex_compressed_decoder (6) -- merged from _draft_streams_compressed_decoder.py
    build_compressed_c1_alu_shift_ca_mux_stream,
    build_compressed_c2_reg_ctrl_mux_stream,
    build_compressed_lui_addi16sp_special_case_stream,
    build_compressed_cl_cs_immediate_scramble_stream,
    build_compressed_ci_addi_li_signext_stream,
    build_compressed_ciw_addi4spn_scramble_stream,
    # ibex_icache (5) -- merged from _draft_streams_icache.py
    build_ibex_icache_enable_toggle_stream,
    build_ibex_icache_fencei_reinvalidate_burst_stream,
    build_ibex_icache_enable_during_invalidate_stream,
    build_ibex_icache_compressed_alignment_skid_stream,
    build_ibex_icache_branch_density_stale_fill_stream,
]

assert len(ALL_STREAM_BUILDERS) == 68, len(ALL_STREAM_BUILDERS)
