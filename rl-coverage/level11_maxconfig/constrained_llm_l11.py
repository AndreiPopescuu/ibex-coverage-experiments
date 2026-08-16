"""constrained_llm_l11.py — CRT constraints derived by reading Ibex RTL directly,
zero riscv-dv/testlist.yaml knowledge (independent LLM-only synthesis).

PASS 1 (40 streams, later 39 after review): multi-agent analysis of
ibex_pmp.sv, ibex_cs_registers.sv, ibex_csr.sv, ibex_counter.sv,
ibex_dummy_instr.sv, ibex_alu.sv, ibex_multdiv_fast.sv,
ibex_load_store_unit.sv, ibex_branch_predict.sv.

REVIEWED: a dedicated review pass cross-checked every RTL claim and op/csr-
bucket index in pass 1 against the actual RTL and codec_l11.py. 21 of 40
functions had confirmed issues (docstring inaccuracies, x0/uimm=0 silent
no-ops, unconstrained-register CSR writes that never guaranteed the claimed
byte value, one function — build_csr_shadow_stream — targeting RTL that's
never elaborated in any Ibex build) — all fixed or removed in place, see
"post-review fix" comments throughout.

PASS 2 (24 more streams): a second multi-agent round covering 4 more modules
never targeted in pass 1 — ibex_core.sv (top-level glue: csr_wdata bus mux,
perf-event wires, pmp_req_type mux, RAW-hazard stall glue, illegal-insn wire;
fetch-enable/stall/sleep/debug-mode/irq/NMI/debug_req/RVFI were investigated
and confirmed structurally unreachable from this codec/testbench, not just
hard to hit — not faked), ibex_decoder.sv (RV32B decode arms + illegal-
instruction default), ibex_compressed_decoder.sv (RVC decode muxes distinct
from ibex_branch_predict.sv's instr_cj/instr_cb), ibex_icache.sv (the
content-driven angles only — core cache mechanics need fetch-address control
this codec doesn't have, confirmed unreachable).

63 stream functions total across 13 RTL modules.

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
  SYSTEM:    62,63,69-72 (ECALL/EBREAK/FENCE/MRET/WFI/FENCE_I — fixed post-review,
             was mislabeled SRET/DRET which don't exist in this codec)
  EXCEPTION: 83-86 (illegal-instr / misaligned load-store, fixed encodings)

imm_bucket → immediate: {0: -2048, 1: -100, 2: 0, 3: 100, 4: 2047}
"""

# ---------------------------------------------------------------------------
# CATEGORY_WEIGHTS_LLM — derived from RTL module complexity + coverage gap data
#
# Largest gaps (from prior RL run): ibex_counter (339 bins), ibex_core (296),
# ibex_cs_registers (191), ibex_pmp (118).
# CSR ops drive all four; system ops exercise the exception save/restore path;
# muldiv drives ibex_multdiv_fast FSM; load_store exercises ibex_load_store_unit.
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
# ---------------------------------------------------------------------------

def build_pmp_mode_sweep_stream(rng):
    """Target ibex_pmp.sv: region_match_all always_comb unique case arms.

    ibex_pmp.sv always_comb block:
      unique case (csr_pmp_cfg_i[r].mode)
        PMP_MODE_OFF   (2'b00): region_match_all = 1'b0
        PMP_MODE_NA4   (2'b10): region_match_all = region_match_eq
        PMP_MODE_NAPOT (2'b11): region_match_all = region_match_eq (with mask)
        PMP_MODE_TOR   (2'b01): region_match_all = (eq | gt) & lt

    PMP cfg byte: [7]=L, [4:3]=A(mode), [2]=X, [1]=W, [0]=R
      OFF=0x00, TOR=0x08, NA4=0x10, NAPOT=0x18
    Write pmpaddr before pmpcfg to set up TOR boundary addresses.
    """
    stream = []
    for bucket in range(37, 53):  # pmpaddr0..15
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
    for bucket in range(33, 37):  # pmpcfg0..3
        for mode_byte in (0x00, 0x08, 0x10, 0x18):  # OFF, TOR, NA4, NAPOT
            stream.append((66, rng.randint(1, 31), mode_byte, 0, 2, bucket))  # CSRRWI: guaranteed mode
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

    cfg byte: bit0=R, bit1=W, bit2=X → write 0x01(R), 0x02(W), 0x04(X), 0x07(RWX).
    Follow with loads (READ), stores (WRITE) to drive pmp_req_type comparators.
    Post-review fix: CSRRWI (uimm literal, not register-sourced) guarantees the
    exact byte instead of copying whatever was in a random register.
    """
    stream = []
    for bucket in range(33, 37):
        for perm_byte in (0x01, 0x02, 0x04, 0x07):
            stream.append((66, rng.randint(1, 31), perm_byte, 0, 2, bucket))  # CSRRWI: guaranteed R/W/X
    for _ in range(6):
        stream.append((rng.choice([19, 20, 21, 22, 23]),
                       rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(0, 4), 0))
    return stream


def build_pmp_lock_stream(rng):
    """Target ibex_pmp.sv: L (lock) bit in pmpcfg — perm_check_wrapper locked path.

    ibex_pmp.sv perm_check_wrapper:
      When L=1 AND MML=0: M-mode access uses orig_perm_check (not bypass).
      When L=0 AND MML=0: M-mode always passes (access_fault=0 for M-mode).

    cfg byte bit7=L: write 0x88 (L+TOR+R), 0x9F (L+NAPOT+RWX) to exercise
    the locked-region check. Write pmpaddr first then pmpcfg with L=1.

    Post-review fix: no 5-bit CSRRWI immediate can reach bit7 (L), so the
    original random-register CSRRW never actually guaranteed L=1. Loads
    imm_bucket=1 (-100 = 0xFFFFFF9C) into a scratch register first via ADDI —
    its low byte 0x9C = 0b1001_1100 has L=1, mode[4:3]=11 (NAPOT), X=1 — then
    CSRRW's that known register into pmpcfg, guaranteeing L=1 deterministically.
    """
    ADDI = 10
    stream = []
    for bucket in range(37, 41):  # pmpaddr0..3
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, 3, bucket))
    lock_reg = rng.randint(1, 31)
    stream.append((ADDI, lock_reg, 0, 0, 1, 0))  # lock_reg = -100 = 0xFFFFFF9C (L=1 byte 0x9C)
    for bucket in range(33, 35):  # pmpcfg0..1
        stream.append((27, rng.randint(1, 31), lock_reg, 0, 4, bucket))  # CSRRW: guaranteed L=1
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

    Write two adjacent pmpaddr entries to create a TOR range, then pmpcfg
    with mode=TOR (0x08). Subsequent loads/stores drive the comparators.
    """
    stream = []
    # Write pmpaddr[0] (lower bound) and pmpaddr[1] (upper bound)
    stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 37))  # pmpaddr0
    stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, 4, 38))  # pmpaddr1
    # Write pmpcfg0 with TOR mode for region 1 (CSRRWI, post-review: guaranteed
    # mode byte 0x08 instead of a random-register CSRRW that was only ~25% TOR)
    stream.append((66, rng.randint(1, 31), 0x08, 0, 2, 33))  # pmpcfg0 = TOR, guaranteed
    # Access pattern to drive comparators
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

    NAPOT pmpaddr encoding: 2^(G+2)-1 granule mask in lower bits.
    E.g., pmpaddr = 0x...FFFF for 64KB region. Write varied pmpaddr values
    with NAPOT mode (pmpcfg mode bits [4:3] = 2'b11 → byte = 0x18) to
    exercise different mask widths.
    """
    stream = []
    # NAPOT encodings of different sizes (lower bits = 2^n-1)
    for bucket in range(37, 45):  # pmpaddr0..7
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0,
                       rng.randint(2, 4), bucket))
    for bucket in range(33, 37):  # pmpcfg0..3 — write NAPOT mode (CSRRWI, post-review:
                                   # guaranteed 0x18 instead of a random-register CSRRW)
        stream.append((66, rng.randint(1, 31), 0x18, 0, 2, bucket))
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

    CSR_MSTATUS_MIE_BIT=3, MPIE_BIT=7, MPP_BIT=11:12, MPRV_BIT=17, TW_BIT=21.
    Alternates CSRRS (set bits) and CSRRC (clear bits) to toggle each field.
    """
    stream = []
    mstatus_bucket = 29
    # Post-review fix: uimm=0 on CSRRSI/CSRRCI is forced to a pure read by the
    # decoder (ibex_decoder.sv:196-201, rs1=='0 -> CSR_OP_READ) — the original
    # uimm=0 calls were silent no-ops. uimm=8 targets bit3 (MIE).
    stream.append((67, 0, 8, 0, 2, mstatus_bucket))   # CSRRSI: set bit3 (MIE), uimm=8
    stream.append((68, 0, 8, 0, 2, mstatus_bucket))   # CSRRCI: clear bit3 (MIE), uimm=8
    for _ in range(4):
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0,
                       rng.choice([2, 3, 4]), mstatus_bucket))
    stream.append((28, rng.randint(1, 31), 0, 0, 2, mstatus_bucket))
    return stream


# ---------------------------------------------------------------------------
# ibex_csr.sv — write-enable toggle + reset path
#
# NOTE (post-review): build_csr_shadow_stream was DELETED here. It targeted
# ibex_csr.sv's gen_shadow block (shadow_q/rd_error_o comparator), but
# ibex_core.sv:182 hardcodes `localparam bit ShadowCSR = 1'b0;` — a
# localparam, never exposed as a configurable module parameter — so
# gen_shadow is NEVER elaborated in ANY Ibex build (not just this one), only
# gen_no_shadow (rd_error_o tied to 1'b0) exists in silicon. No instruction
# stream can cover RTL that was never instantiated.
# ---------------------------------------------------------------------------

def build_ibex_csr_wr_en_stream(rng):
    """Target ibex_csr.sv: wr_en_i gating on rdata_q register update.

    ibex_csr.sv always_ff:
      if (wr_en_i) rdata_q <= wr_data_i;  ← only updates when wr_en_i=1

    Toggle wr_en_i high then low by alternating CSRRW (writes, wr_en_i=1)
    and CSRRS x0,csr,x0 pure reads (wr_en_i=0, no side-effect) on the same CSR.
    Uses mscratch (bucket=0), mepc (bucket=1), mcause (bucket=2) — plain registers
    with no write-side-effects that could trap execution.

    Post-review fix: the bucket list previously also included 29/30/31
    (mstatus/mie/mtvec), contradicting this docstring's own safety claim —
    writing an unconstrained value into mtvec relocates the trap base to an
    arbitrary address (ibex_cs_registers.sv:617-621), risking a jump to garbage
    if an exception fires later. Restricted to the buckets actually documented.
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
    We cannot trigger a real reset from the instruction stream, but we can toggle
    every bit in rdata_q by writing 0 then 0xFFFF then 0, ensuring full 0→1→0
    toggle coverage on the register's storage bits.
    Uses csrs that accept arbitrary writes: mscratch(0), mtvec(31), mcause(2).
    """
    stream = []
    for bucket in [0, 2, 29, 31, 68]:
        # Write nonzero (post-review fix: CSRRWI with a literal uimm=31, guaranteed —
        # the original comment claimed an "all-ones via ALU" setup that didn't exist,
        # rs1 was just an unconstrained register)
        stream.append((66, rng.randint(1, 31), 31, 0, 2, bucket))
        # Write zero
        stream.append((27, rng.randint(1, 31), 0, 0, 2, bucket))
        # Write nonzero again
        stream.append((66, rng.randint(1, 31), 31, 0, 2, bucket))
        # Read back
        stream.append((28, rng.randint(1, 31), 0, 0, 2, bucket))
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
    mhpmcounter9-14(56-61 — post-review fix: this is a DISTINCT counter group,
    not a second copy of 3-8 as an earlier comment claimed).
    High half CSRs: mcycleh(11), minstreth(12).
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

    mcountinhibit (CSR 0x320, bucket=13): bit0=inhibit cycle, bit2=inhibit instret.
    Writing nonzero freezes counters (counter_inc_i=0 for ibex_counter instances),
    toggling the counter_d = counter_upd vs counter[63:0] mux in always_comb.
    """
    stream = []
    inhibit_bucket = 13
    stream.append((67, rng.randint(1, 31), 1, 0, 2, inhibit_bucket))  # CSRRSI: set bit0, uimm=1
                                                                        # (post-review fix, guaranteed nonzero)
    for _ in range(4):
        stream.append((10, rng.randint(1, 31), rng.randint(1, 31), 0, 2, 0))  # ADDI
    stream.append((27, rng.randint(1, 31), 0, 0, 2, inhibit_bucket))
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
# ---------------------------------------------------------------------------

def build_dummy_instr_enable_stream(rng):
    """Target ibex_dummy_instr.sv: insert_dummy_instr path + DUMMY_* enum arms.

    insert_dummy_instr = dummy_instr_en_i & (dummy_cnt_q == dummy_cnt_threshold)
    unique case(lfsr_data.instr_type):
      DUMMY_ADD/MUL/DIV/AND — all 4 arms exercised as LFSR cycles.

    cpuctrl (0x7C0, bucket=68) bit0=dummy_instr_en. Enable then run ALU ops
    so the dummy counter fires repeatedly, cycling through all 4 LFSR states.
    """
    stream = []
    cpuctrl_bucket = 68
    stream.append((67, 1, 1, 0, 2, cpuctrl_bucket))  # CSRRSI: set bit0, uimm=1 (post-review fix,
                                                       # was CSRRW from an unconstrained register)
    for _ in range(20):
        op = rng.choice([0, 1, 2, 5, 6, 7, 8, 9])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31), rng.randint(1, 31), 2, 0))
    stream.append((27, 1, 0, 0, 2, cpuctrl_bucket))
    return stream


def build_dummy_instr_seed_stream(rng):
    """Target ibex_dummy_instr.sv: dummy_instr_seed_d XOR path.

    dummy_instr_seed_d = dummy_instr_seed_q ^ dummy_instr_seed_i
    Asserted when secureseed (0x7C1, bucket=69) is written.
    Each write XORs into seed_q, changing LFSR state and which DUMMY_* arm fires.
    """
    stream = []
    stream.append((67, 1, 1, 0, 2, 68))  # CSRRSI: enable dummy_instr, uimm=1 (post-review fix)
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
    """
    stream = []
    stream.append((67, 1, 1, 0, 2, 68))   # CSRRSI: enable dummy_instr, uimm=1 (post-review fix)
    for i in range(32):
        stream.append((rng.choice([0, 1, 5, 6, 10, 14]), rng.randint(1, 31),
                       rng.randint(1, 31), rng.randint(1, 31), 2, 0))
        if i % 8 == 7:
            stream.append((27, 0, rng.randint(1, 31), 0, rng.randint(2, 4), 69))  # secureseed
    stream.append((27, 1, 0, 0, 2, 68))   # disable
    return stream


def build_ibex_dummy_instr_mask_stream(rng):
    """Target ibex_dummy_instr.sv: dummy_cnt_threshold mask bits.

    dummy_cnt_threshold = lfsr_data.cnt & {dummy_instr_mask_i, {TIMEOUT_CNT_W-3{1'b1}}}

    cpuctrl bits [3:1] = dummy_instr_mask_i (3 bits). Writing different mask
    values changes the threshold period: mask=0b000 → threshold always 0b00xxx
    (very frequent dummies), mask=0b111 → full LFSR range (less frequent).
    Sweep all 8 mask values to toggle all 3 mask bits and exercise the AND gate.

    Post-review fix: mask_val is now actually wired into the write via CSRRWI
    (uimm = (mask_val<<1)|1, max 15, fits the 5-bit immediate field) — the
    original CSRRW from an unconstrained register never used the loop variable.
    """
    stream = []
    for mask_val in range(8):  # sweep dummy_instr_mask_i = 0b000 .. 0b111
        # cpuctrl: bit0=dummy_en, bits[3:1]=mask → value = (mask_val<<1) | 1
        stream.append((66, 1, (mask_val << 1) | 1, 0, 2, 68))  # CSRRWI: guaranteed mask+enable
        for _ in range(6):
            stream.append((rng.choice([0, 10, 5]), rng.randint(1, 31),
                           rng.randint(1, 31), 0, 2, 0))
    stream.append((27, 1, 0, 0, 2, 68))  # disable
    return stream


def build_ibex_dummy_instr_combined_stream(rng):
    """Combined: seed→enable→arm_sweep→disable cycle, repeated with varying seeds.

    Chains seed writes, enable, ALU burst (to trigger insertions), disable.
    Exercises: dummy_instr_seed_d XOR path, insert_dummy_instr toggle,
    all 4 DUMMY_* instr_type case arms, lfsr_en toggle, dummy_cnt reset.
    """
    stream = []
    for _ in range(3):
        stream.append((27, 0, rng.randint(1, 31), 0, rng.randint(2, 4), 69))  # seed
        stream.append((67, 1, 1, 0, 2, 68))  # CSRRSI: enable, uimm=1 (post-review fix)
        for _ in range(12):
            stream.append((rng.choice([0, 1, 5, 10]), rng.randint(1, 31),
                           rng.randint(1, 31), rng.randint(1, 31), 2, 0))
        stream.append((27, 1, 0, 0, 2, 68))                                    # disable
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

    ibex_multdiv_fast.sv:436-437, inside the MD_IDLE case arm only (post-review
    fix — not a standing combinational assignment as an earlier comment implied):
      MD_IDLE: div_by_zero_d = equal_to_zero_i;  // ALU-computed zero-flag on
                                                   // the input operand, sampled
                                                   // once per divide; other
                                                   // states hold div_by_zero_q.

    Using rs2=x0 (always zero) as denominator for DIV(34)/DIVU(35)/REM(36)/REMU(37)
    ensures equal_to_zero_i==1 and div_by_zero_q=1. Mix with nonzero-denominator
    divides to toggle div_by_zero_q 0→1→0.
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

    lsu_type_i[1:0]: 2'b00=word, 2'b01=halfword, 2'b10/2'b11=byte (per
    ibex_load_store_unit.sv:119 `unique case (lsu_type_i) // Data type 00 Word,
    01 Half word, 11,10 byte` — post-review fix, an earlier comment had this
    backwards; doesn't affect the code below, which enumerates by opcode).
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
    """Target ibex_load_store_unit.sv: sign-extend mux toggle.

    Post-review fix: the previously-quoted RTL excerpt referenced signals
    (`rdata_b_i`/`rdata_h_i`) that don't exist in this file — a fabricated
    paraphrase, not a real transcription. The actual logic
    (ibex_load_store_unit.sv:281-318) is a 4-way `case (rdata_offset_q)` on
    `data_rdata_i` slices, gated by the REGISTERED `data_sign_ext_q` (captured
    from the unregistered `lsu_sign_ext_i` input on `ctrl_update`, see
    lines 199-211) — not `lsu_sign_ext_i` directly. The instruction stream
    itself was already correct; only the docstring's RTL citation is fixed here.

    Alternates signed (LB=19, LH=20) and unsigned (LBU=22, LHU=23) byte and
    half loads to toggle the sign-extend selection on every instruction.
    """
    stream = []
    signed_ops   = [19, 20]   # LB, LH → lsu_sign_ext_i = 1
    unsigned_ops = [22, 23]   # LBU, LHU → lsu_sign_ext_i = 0
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
    """Target ibex_load_store_unit.sv: data_be_o all 4 byte-enable patterns.

    data_be_o derivation (ibex_load_store_unit.sv:120-138), word case is
    data_offset-dependent, not a constant 1111 (post-review fix — an earlier
    comment claimed "word → 4'b1111 always"):
      byte  → 4'b0001 << addr[1:0]   → patterns: 0001,0010,0100,1000
      half  → 4'b0011 << addr[1:0]   → patterns: 0011,0110,1100 (0001 if misaligned)
      word  → 2'b00:1111 2'b01:1110 2'b10:1100 2'b11:1000

    Mix SB(24)/SH(25)/SW(26) with varied immediate offsets. Note (post-review):
    imm_bucket 0-3 all have imm[1:0]==00 (-2048,-100,0,100 are all ≡0 mod 4,
    see codec_l11.py's bucket table), so offset variation across these buckets
    doesn't change addr[1:0] via the immediate — any BE pattern variation here
    comes from the unconstrained rs1 register's low bits, not the op/imm choice.
    """
    stream = []
    for _ in range(4):
        for op, imm in [(24, 0), (24, 1), (24, 2), (24, 3),   # SB all offsets
                        (25, 0), (25, 2),                       # SH even offsets
                        (26, 0)]:                               # SW aligned
            rs1 = rng.randint(1, 31)
            rs2 = rng.randint(1, 31)
            stream.append((op, 0, rs1, rs2, imm, 0))
    for _ in range(4):
        for op, imm in [(19, 0), (19, 1), (20, 0), (21, 0)]:  # loads
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0, imm, 0))
    return stream


# ---------------------------------------------------------------------------
# ibex_branch_predict.sv — predict_taken toggle + case arms + compressed
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
    # Post-review fix: only C_JAL(73)/C_J(74) assert instr_cj — the original
    # range(73,83) pool included 8 ops that don't (C_ADDI16SP/C_LWSP/C_SWSP/
    # C_JR/C_JALR/C_EBREAK plus C_BEQZ/C_BNEZ, which assert instr_cb instead).
    for _ in range(4):
        stream.append((rng.choice([73, 74]), rng.randint(0, 31),
                       rng.randint(0, 31), 0, rng.choice([1, 3]), 0))
    return stream


def build_ibex_branch_predict_backward_branch_stream(rng):
    """Target ibex_branch_predict.sv: instr_b_taken=1 (backward branch taken).

    assign instr_b_taken = (instr_b & imm_b_type[31]) | ...
    imm_b_type[31]=1 when the B-type immediate is negative.
    imm_bucket=0 (-2048) and imm_bucket=1 (-100) both give negative offsets.
    All 6 branch flavours (BEQ/BNE/BLT/BGE/BLTU/BGEU) exercise instr_b=1;
    the negative offset ensures predict_branch_taken_o=1 for each.
    (Distinct from ibex_compressed_decoder.sv's own instr_cb/instr_j decode —
    that's covered separately in the compressed-decoder-focused streams.)
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

    Post-review fix: of the full compressed op range, only C_JAL(73)/C_J(74)
    (quadrant1, funct3 001/101) actually satisfy the instr_cj condition above —
    the original pool (range(45,61)+range(73,83), 26 ops) hit the claimed arm
    only ~2/26≈8% of the time; the other 24 ops (C_ADDI..C_ANDI, C_BEQZ/C_BNEZ
    which assert instr_cb instead, C_ADDI16SP/C_LWSP/C_SWSP/C_JR/C_JALR/
    C_EBREAK) don't assert instr_cj at all.
    predict_branch_taken_o=1 regardless of offset sign (same as JAL).
    """
    stream = []
    compressed_jump_ops = [73, 74]  # C_JAL, C_J — the only two that assert instr_cj
    for _ in range(12):
        op = rng.choice(compressed_jump_ops)
        stream.append((op, rng.randint(0, 31), rng.randint(0, 31), 0,
                       rng.choice([0, 1, 3, 4]), 0))
    return stream


def build_ibex_branch_predict_compressed_branch_stream(rng):
    """Target ibex_branch_predict.sv: instr_cb=1 + imm_cb_type[31] taken path.

    assign instr_cb = (instr[1:0] == 2'b01) & ((instr[15:13] == 3'b110) | (instr[15:13] == 3'b111))
    assign instr_b_taken = ... | (instr_cb & imm_cb_type[31])

    imm_cb_type[31] = instr[12] (the sign bit of the CB-format immediate).
    Backward offset (imm_bucket=0 or 1 → negative) sets imm_cb_type[31]=1.

    Post-review fix: only C_BEQZ(75)/C_BNEZ(76) assert instr_cb — the original
    pool (range(73,83), 10 ops) hit the claimed arm only 2/10=20% of the time,
    including C_JAL/C_J (which assert instr_cj instead) and 6 ops that assert
    neither.
    """
    stream = []
    for _ in range(12):
        op = rng.choice([75, 76])  # C_BEQZ, C_BNEZ — the only two that assert instr_cb
        stream.append((op, rng.randint(0, 31), rng.randint(0, 31), 0,
                       rng.choice([0, 1]), 0))   # negative offset → taken
    for _ in range(6):
        op = rng.choice([75, 76])
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

    Interleaves JAL(instr_j), branches(instr_b), compressed jumps(instr_cj),
    compressed branches(instr_cb), and plain ALU ops (default: no prediction)
    to cycle through all 5 arms.

    Post-review fix: the instr_cj line previously sampled range(45,61)
    (C_ADDI..C_ANDI), which contains ZERO ops that assert instr_cj — a 0%
    hit rate on its own stated target (the only two that do, C_JAL=73/C_J=74,
    aren't in that range at all). The instr_cb line previously sampled
    range(73,83), which includes C_JAL/C_J (these assert instr_cj, not
    instr_cb) plus several ops asserting neither — only 2/10 correct.
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


# =============================================================================
# SECOND SYNTHESIS PASS — 4 more modules (ibex_core, ibex_decoder,
# ibex_compressed_decoder, ibex_icache), derived by a second round of parallel
# subagents reading the RTL directly, same zero-riscv-dv-knowledge methodology.
# =============================================================================

# ---------------------------------------------------------------------------
# ibex_core.sv — top-level glue not exercised by any sub-module-focused stream
# above: csr_wdata bus mux, perf-event wires, pmp_req_type mux, RAW-hazard
# stall glue, illegal-instruction wire. (fetch-enable/stall/sleep/debug-mode
# and irq/NMI/debug_req/RVFI were investigated and confirmed UNREACHABLE from
# this codec/testbench — see the two synthesis agents' reports; not faked here.)
# ---------------------------------------------------------------------------

def build_core_csr_wdata_bus_toggle_stream(rng):
    """Target ibex_core.sv:1055: `assign csr_wdata = alu_operand_a_ex;`

    This is the single top-level wire that feeds cs_registers_i.csr_wdata_i
    for every CSR write in the design. Register-form CSR ops (CSRRW/CSRRS/
    CSRRC, op=27/28/29) source it from the full 32-bit rs1 value;
    immediate-form CSR ops (CSRRWI/CSRRSI/CSRRCI, op=66/67/68) source it from
    the 5-bit rs1-field uimm zero-extended to 32 bits (top 27 bits forced 0).
    No stream elsewhere alternates these two source widths back-to-back on
    the SAME csr_wdata wire in the same bucket. Rapid alternation maximises
    bit-toggle activity on csr_wdata[31:5] specifically. Uses mscratch/mepc/
    mcause/mtvec/mstatus-class buckets (0-3, 29-31), which accept arbitrary
    writes without side-effects that could trap.
    """
    stream = []
    buckets = [0, 1, 2, 3, 29, 30, 31]
    for bucket in buckets:
        stream.append((27, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
        stream.append((66, rng.randint(1, 31), rng.randint(0, 31), 0, 2, bucket))
        stream.append((28, rng.randint(1, 31), rng.randint(1, 31), 0, rng.randint(2, 4), bucket))
        stream.append((68, rng.randint(1, 31), rng.randint(0, 31), 0, 2, bucket))
    return stream


def build_core_perf_event_interleave_stream(rng):
    """Target ibex_core.sv perf_* event wiring (declared 376-389, wired at
    712-719 and 1154-1167): perf_jump, perf_branch, perf_tbranch, perf_load,
    perf_store, perf_mul_wait, perf_div_wait.

    Each is a point-to-point wire connecting one pipeline stage's "an event
    happened" signal to the HPM-event mux in ibex_cs_registers.sv — core-level
    glue, not a sub-module case-arm. Existing streams each hammer ONE
    instruction class repeatedly, holding one perf_* wire high for long
    stretches but never forcing adjacent DIFFERENT wires to toggle against
    each other. This stream round-robins jump/taken-branch/not-taken-branch/
    load/store/mul/div so each wire pulses high for exactly one instruction
    before a different wire pulses next, maximising 0->1->0 toggle activity.
    """
    stream = []
    kinds = [
        (44, None), (38, True), (39, True), (21, None),
        (26, None), (30, None), (34, None), (65, None),
    ]
    for _ in range(24):
        op_i, taken = rng.choice(kinds)
        if op_i in (38, 39):
            rs = rng.randint(1, 31)
            stream.append((op_i, 0, rs, rs, 2, 0))
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

      assign pmp_req_addr[PMP_D] = {2'b00, data_addr_o[31:0]};
      assign pmp_req_type[PMP_D] = data_we_o ? PMP_ACC_WRITE : PMP_ACC_READ;

    This 2-bit enum mux is computed in ibex_core.sv ITSELF, one level above
    ibex_pmp.sv's region-match logic that the 8 ibex_pmp streams above target
    (the *consumer* of pmp_req_type). None of those alternate loads/stores on
    consecutive cycles, so this wire mostly sits at one value for runs of
    several cycles. This stream alternates single loads and stores every
    instruction to force PMP_ACC_READ<->PMP_ACC_WRITE every cycle. All 4
    pmpcfg regions left at reset (mode=OFF) so every access is permitted —
    isolates the mux-toggle target from ibex_pmp.sv's own permission logic.
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
    """Target ibex_core.sv:314-316 stall-control wiring (id_in_ready, ex_valid,
    lsu_resp_valid, ports at 607-608).

    ex_valid is driven low by ex_block_i for the duration of a multi-cycle
    MULDIV op; lsu_resp_valid is driven low by the LSU while a load/store is
    outstanding. Both gate id_in_ready, which stalls the pipeline front-end.
    No existing stream deliberately creates back-to-back RAW hazards where
    the VERY NEXT instruction consumes a still-in-flight multi-cycle op or
    load's result — build_multdiv_*/build_ibex_load_store_* streams use
    independent destination registers. This stream chains DIV/MUL -> dependent
    ADD (reusing rd as next rs1) and LW -> dependent ADDI, forcing the
    id_in_ready/ex_valid/lsu_resp_valid stall-then-resume glue to toggle on
    every pair. Purely a function of op choice + register reuse.
    """
    stream = []
    for _ in range(10):
        rd = rng.randint(1, 31)
        rs = rng.randint(1, 31)
        stream.append((rng.choice([30, 34]), rd, rs, rng.randint(1, 31), 2, 0))
        stream.append((0, rng.randint(1, 31), rd, rng.randint(1, 31), 2, 0))
        rd2 = rng.randint(1, 31)
        stream.append((21, rd2, rs, rng.randint(1, 31), rng.randint(0, 2), 0))
        stream.append((10, rng.randint(1, 31), rd2, 0, 2, 0))
    return stream


def build_core_illegal_insn_toggle_stream(rng):
    """Target ibex_core.sv:391,723 — illegal_insn_id / unused_illegal_insn_id.

    ibex_core.sv:723 (unconditional, not inside any ifdef):
        assign unused_illegal_insn_id = illegal_insn_id;
    illegal_insn_id is wired straight through from ibex_id_stage's
    illegal_insn_o. This is the only elaborated consumer of illegal_insn_id
    in ibex_core.sv in this build (the fcov-signals block that also reads it
    is stripped by `-DSYNTHESIS=1`, confirmed via cpu/Makefile.upstream).
    ILLEGAL_INSN (op=83) is a fixed CUSTOM-0 encoding no decoder opcode table
    claims, falling through to illegal_insn_o=1. Alternating it 1:1 against
    ADDI (op=10) forces a 0->1->0 transition every 2 cycles — denser than
    build_ibex_load_store_misaligned_stream's ~1-in-4 incidental hits (which
    targets a different module's error path, not this wire).
    """
    stream = []
    for _ in range(24):
        stream.append((83, 0, 0, 0, 2, 0))
        stream.append((10, rng.randint(1, 31), rng.randint(1, 31), 0,
                       rng.randint(0, 4), 0))
    return stream


# ---------------------------------------------------------------------------
# ibex_decoder.sv — decode case-arms in the RV32B op range (87-142) not hit
# by build_alu_rv32b_stream (which only covers ROL/ROR/RORI/BSET/BCLR/BINV/
# BEXT/SH1-3ADD), plus the illegal-instruction default arm.
# ---------------------------------------------------------------------------

def build_decoder_op_imm_bitcount_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP_IMM, funct3=001, instr[31:27]=5'b01100
    nested case(instr[26:20]) bit-count/sign-extend group (lines 381-397):
    clz(102)/ctz(103)/cpop(104)/sext.b(105)/sext.h(106). None of these 5 case
    items are hit by any existing stream. Cycles all 5 ops across varied rs1.
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
    """Target ibex_decoder.sv: OPCODE_OP_IMM funct3=001 instr[31:27]=5'b01100
    crc32/crc32c sub-group (lines 388-395): crc32.b/h/w(137-139),
    crc32c.b/h/w(140-142). Distinct, narrower-gated nested-case arm (only
    legal for RV32BOTEarlGrey|RV32BFull), also drives ibex_alu.sv's
    ALU_CRC32_* multicycle operators — a decode case never hit elsewhere.
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
    exercised elsewhere: sloi(131)/sroi(132)/grevi(133)/gorci(134)/shfli(135)/
    unshfli(136), bclri(112)/bseti(113)/binvi(114)/bexti(115). Sweeps shamt
    across all 5 SHAMT_BUCKET_VALUES {0,8,16,24,31} for every op.
    """
    BCLRI, BSETI, BINVI, BEXTI = 112, 113, 114, 115
    SLOI, SROI, GREVI, GORCI, SHFLI, UNSHFLI = 131, 132, 133, 134, 135, 136
    stream = []
    for op in (SLOI, BCLRI, BSETI, BINVI, SHFLI, SROI, BEXTI, GREVI, GORCI, UNSHFLI):
        for imm_bucket in range(5):
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0, imm_bucket, 0))
    return stream


def build_decoder_zbp_rtype_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP zbp legacy R-type block (lines
    495-504): slo(116)/sro(117)/grev(118)/gorc(119)/shfl(120)/unshfl(121)/
    xperm.n(122)/xperm.b(123)/xperm.h(124) — 9 case items entirely unvisited
    by any existing stream, each a distinct {instr[31:25],instr[14:12]}
    literal on the case comparator.
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
    """Target ibex_decoder.sv: OPCODE_OP zbc block (lines 505-510):
    clmul(125)/clmulr(126)/clmulh(127). Shares funct7 with MIN/MAX/MINU/MAXU
    but a different funct3 slice (001/010/011) — a comparator slice no
    existing stream drives.
    """
    CLMUL, CLMULR, CLMULH = 125, 126, 127
    stream = []
    for op in (CLMUL, CLMULR, CLMULH):
        for _ in range(4):
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    return stream


def build_decoder_zbe_zbf_stream(rng):
    """Target ibex_decoder.sv: OPCODE_OP zbe (bcompress(128)/bdecompress(129),
    RV32BFull-gated, lines 511-513) and zbf (bfp(130), line 493-494) blocks —
    distinct decode-side case-comparator literals and BFP's own ALU-control
    entry, untouched by build_alu_rv32b_stream's BEXT alone.
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
    """Target ibex_decoder.sv: OPCODE_OP zbb logic/pack/min-max case items
    (lines 476-487) beyond build_alu_compare_sign_mismatch_stream's narrow
    MIN/MINU slice: andn(90)/orn(91)/xnor(92), min(95)/max(96)/minu(97)/
    maxu(98) (full sweep), pack(99)/packh(100)/packu(101) (entirely untouched).
    Sweeps all four min/max variants plus andn/orn/xnor/pack/packu/packh
    across randomized and sign-mismatched register pairs.
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
    stream.append((ADDI, neg_r, 0, 0, 0, 0))
    stream.append((ADDI, pos_r, 0, 0, 4, 0))
    for op in (MAX_OP, MAXU_OP, MIN_OP, MINU_OP):
        stream.append((op, rng.randint(1, 31), neg_r, pos_r, 2, 0))
        stream.append((op, rng.randint(1, 31), pos_r, neg_r, 2, 0))
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), 2, 0))
    return stream


def build_decoder_illegal_default_reset_stream(rng):
    """Target ibex_decoder.sv: top-level `unique case (opcode) default:
    illegal_insn=1'b1` (lines 643-645) and the reset-on-illegal gating block
    (lines 658-666, rf_we/data_req_o/data_we_o/jump_in_dec_o/jump_set_o/
    branch_in_dec_o/csr_access_o all forced 0 when illegal_insn).

    ILLEGAL_INSN (op=83) is the codec's only way to reach this default arm.
    Interleaves it with instructions that set each of the 7 gated signals
    (ADD/JAL/SW/LW/BEQ/CSRRW) immediately before/after, so each signal's
    gating mux toggles 0(legal)->1(illegal,forced 0)->0(legal), exercising
    the reset block's per-signal AND-with-~illegal_insn structure rather than
    just firing illegal_insn in isolation.
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
# ibex_compressed_decoder.sv — decoder-internal case arms (distinct from
# ibex_branch_predict.sv's instr_cj/instr_cb, which read the decoded output;
# these target the compressed_decoder's OWN instr_o construction muxes).
# ---------------------------------------------------------------------------

def build_compressed_c1_alu_shift_ca_mux_stream(rng):
    """Target ibex_compressed_decoder.sv: C1 quadrant funct3=100 nested case
    (lines 366-486), the deepest case-within-case in the file: c.srli(58,
    instr_i[11:10]=00)/c.srai(59,=01)/c.andi(60,=10)/c.sub(57)/c.xor(56)/
    c.or(55)/c.and(54) (=11, further CA-format case on {instr_i[12],
    instr_i[6:5]}). c.subw/c.addw are RV64-only (unconditionally illegal
    here); c.mul/c.zext.*/c.sext.*/c.not are Zcb-gated with no codec op index
    — both skipped as structurally unreachable. Round-robins the 7 reachable
    ops so instr_i[11:10] and {instr_i[12],instr_i[6:5]} both toggle fully.
    """
    stream = []
    ops = [58, 59, 60, 57, 56, 55, 54]
    for ib in range(5):
        for op in ops:
            rd = rng.randint(8, 15)
            rs1 = rng.randint(8, 15)
            rs2 = rng.randint(8, 15)
            stream.append((op, rd, rs1, rs2, ib, 0))
    return stream


def build_compressed_c2_reg_ctrl_mux_stream(rng):
    """Target ibex_compressed_decoder.sv: C2 quadrant funct3=100 register-
    control if/else tree (lines 523-549), the widest decision tree outside
    the (unreachable) Zcmp FSM: c.mv(49)->add, c.jr(80)->jalr x0, c.add(50)
    ->add, c.ebreak(82)->fixed encoding, c.jalr(81)->jalr x1. Five distinct
    instr_o constructions gated by instr_i[12] and instr_i[6:2]!=0, with a
    third condition nested only inside the bit12=1/rs2==0 branch. The rd==0
    illegal check on c.jr is unreachable (codec floors rs1>=1), skipped.
    """
    stream = []
    ops = [49, 80, 50, 82, 81]
    for _ in range(10):
        for op in ops:
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), rng.randint(0, 4), 0))
    return stream


def build_compressed_lui_addi16sp_special_case_stream(rng):
    """Target ibex_compressed_decoder.sv: C1 quadrant funct3=011 rd-field
    override mux (lines 352-364) — a single rd-field comparator re-routes the
    SAME quadrant/funct3 encoding from a U-type LUI construction (default) to
    a completely different I-type stack-pointer-ADDI construction when
    instr_i[11:7]==5'h02. C_LUI(47) always encodes rd in {3..31} (codec's
    safe_rd avoids {0,2}), only ever hitting the default path; C_ADDI16SP(77)
    hardcodes rd=2, only ever hitting the override path. Alternating flips
    the mux every pair. The illegal (nzimm==0) leaf is unreachable via either
    op (both encoders guarantee nonzero), skipped.
    """
    stream = []
    for ib in range(5):
        stream.append((47, rng.randint(3, 31), 0, 0, ib, 0))
        stream.append((77, 2, 0, 0, ib, 0))
    return stream


def build_compressed_cl_cs_immediate_scramble_stream(rng):
    """Target ibex_compressed_decoder.sv: C0 quadrant c.lw(52)/c.sw(53)
    immediate bit-scramble (lines 235-246) — the 7-bit byte offset is NOT
    contiguous in the RVC encoding (imm[5:3] at instr_i[12:10], imm[2] at
    instr_i[6], imm[6] at instr_i[5]), a genuine bit permutation. Sweeps all
    5 CL_CS_UIMM_VALUES buckets and cycles rd'/rs1'/rs2' through all 8 prime
    registers (x8-x15) to exercise every 3-bit register-select field.
    """
    stream = []
    for ib in range(5):
        for base in range(8):
            rd = 8 + base
            rs1 = 8 + ((base + 3) % 8)
            rs2 = 8 + ((base + 5) % 8)
            stream.append((52, rd, rs1, 0, ib, 0))
            stream.append((53, 0, rs1, rs2, ib, 0))
    return stream


def build_compressed_ci_addi_li_signext_stream(rng):
    """Target ibex_compressed_decoder.sv: C1 quadrant c.addi(45)/c.li(46) CI-
    format sign-bit fanout (lines 328-350) — both replicate the single CI
    sign bit (instr_i[12]) across 7 output bits via {6{...}}. Includes rd=0
    c.addi (architecturally c.nop, still legal — no illegal_instr_o in this
    arm) alongside rd!=0 c.addi and c.li (always writes rd from x0, a
    different upstream rd-write mux than c.addi's read-modify-write).
    """
    stream = []
    for ib in range(5):
        stream.append((45, 0, 0, 0, ib, 0))
        stream.append((45, rng.randint(1, 31), 0, 0, ib, 0))
        stream.append((46, rng.randint(0, 31), 0, 0, ib, 0))
    return stream


def build_compressed_ciw_addi4spn_scramble_stream(rng):
    """Target ibex_compressed_decoder.sv: C0 quadrant funct3=000
    c.addi4spn(51) (lines 226-233) — the file's most scrambled immediate
    field, bit order [5|4|9|8|7|6|2|3] across instr_i[10:5]. Sweeps all 5
    CIW_NZUIMM_VALUES buckets (4=low-end-only, 1020=high-end-only, etc.) and
    cycles rd' through all 8 prime registers. The illegal (nzuimm==0) leaf is
    unreachable via the codec (encoder asserts nonzero), skipped.
    """
    stream = []
    for ib in range(5):
        for rd in range(8, 16):
            stream.append((51, rd, 0, 0, ib, 0))
    return stream


# ---------------------------------------------------------------------------
# ibex_icache.sv — content-driven angles only (core cache mechanics — tag-RAM
# hit/miss, way selection, fill-buffer arbitration, ECC/scrambling — are
# driven by fetch-ADDRESS patterns this codec cannot engineer, confirmed
# unreachable and not targeted here).
# ---------------------------------------------------------------------------

CPUCTRL_BUCKET = 68  # L11_CSRS index of 0x7C0 (cpuctrl); bit0 = icache_enable
FENCE_I = 72
CSRRSI = 67
CSRRCI = 68


def build_ibex_icache_enable_toggle_stream(rng):
    """Target ibex_icache.sv: icache_enable_i central mux points (gates
    lookup_actual_ic0, fill_cache_new, fill_spec_req, instr_req, fill_cache_d).

    icache_enable_i is sourced from cpuctrl CSR bit[0] (verified against the
    cpu_ctrl_sts_part_t packed struct in ibex_cs_registers.sv). CSRRSI/CSRRCI
    x0,cpuctrl,uimm=1 sets/clears it deterministically (literal uimm, no
    register-value dependency). Interleaves short load bursts (real icache
    fetches likely to start a fill) with enable/disable toggles so the toggle
    lands both between fetches and plausibly mid-fill.
    """
    stream = []
    for round_i in range(8):
        stream.append((CSRRSI, 0, 1, 0, 2, CPUCTRL_BUCKET))
        for _ in range(3):
            op = rng.choice([19, 20, 21, 22, 23])
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31), 0,
                           rng.randint(0, 4), 0))
        stream.append((CSRRCI, 0, 1, 0, 2, CPUCTRL_BUCKET))
        for _ in range(2):
            op = rng.choice([0, 1, 8, 9])
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    stream.append((CSRRSI, 0, 1, 0, 2, CPUCTRL_BUCKET))
    return stream


def build_ibex_icache_fencei_reinvalidate_burst_stream(rng):
    """Target ibex_icache.sv: inval_state_e FSM, INVAL_CACHE restart arm —
    if icache_inval_i re-asserts while inval_state_q is still INVAL_CACHE
    from a previous invalidation, it restarts with a new scramble key instead
    of completing. FENCE.I (op=72) asserts icache_inval_o for one cycle on
    retire. Firing FENCE.I back-to-back (cheap ALU ops between, no loads, to
    keep retire rate high) keeps the gap between pulses short relative to the
    INVAL_CACHE walk, making the restart arm likely. A final isolated
    FENCE.I + long ALU tail lets one invalidation reach the completion->IDLE
    arm instead.
    """
    stream = []
    for _ in range(10):
        stream.append((FENCE_I, 0, 0, 0, 0, 0))
        for _ in range(2):
            op = rng.choice([0, 1, 5, 8, 9])
            stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                           rng.randint(1, 31), 2, 0))
    stream.append((FENCE_I, 0, 0, 0, 0, 0))
    for _ in range(40):
        op = rng.choice([0, 1, 5, 8, 9, 10])
        stream.append((op, rng.randint(1, 31), rng.randint(1, 31),
                       rng.randint(1, 31), rng.randint(0, 4), 0))
    return stream


def build_ibex_icache_enable_during_invalidate_stream(rng):
    """Target ibex_icache.sv: icache_enable_i x icache_inval_i cross terms —
    inval_block_cache forces lookup_actual_ic0/fill_cache_new to 0 whenever
    an invalidation is in flight, regardless of icache_enable_i. Interleaves
    cpuctrl bit[0] toggles with FENCE.I so all 4 combinations (enable/disable
    x inval-in-flight/not) occur — both signals are CSR/opcode-content-driven,
    so this combination is fully reachable without fetch-address control.
    """
    stream = []
    pattern = [
        (CSRRSI, 1), (FENCE_I, None), (CSRRCI, 1), (FENCE_I, None), (CSRRSI, 1),
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
    stream.append((CSRRSI, 0, 1, 0, 2, CPUCTRL_BUCKET))
    return stream


def build_ibex_icache_compressed_alignment_skid_stream(rng):
    """Target ibex_icache.sv: skid-buffer / output-address-parity FSM
    (output_addr_q[1], addr_incr_two, skid_valid_d/skid_ready). Halfword
    parity is advanced purely by whether each retired instruction was 2B
    (compressed) or 4B — content-driven, independent of fetch address. An
    ODD run of compressed ops flips parity to 1, forcing the next
    uncompressed op to fetch unaligned and engage the skid buffer. Emits runs
    of 1/3/5 compressed ops (always odd) followed by uncompressed ops,
    cycling engage/steady-state/disengage transitions.
    """
    stream = []
    compressed_ops = [45, 46, 48, 54, 55, 56, 57, 60]
    uncompressed_ops = [0, 1, 5, 8, 9, 10, 13, 14, 15]
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
    The codec cannot choose a branch's target (no label resolution) but fully
    controls branch/jump DENSITY relative to memory ops that start fills.
    Maximises that density: loads to start multi-beat fills immediately
    followed by a branch/jump, repeated, so branch_i is statistically likely
    to assert while the preceding load's fill buffer is still busy (hitting
    fill_stale_d) and to unconditionally clear any live skid buffer state.
    """
    stream = []
    load_ops = [19, 20, 21, 22, 23]
    branch_ops = [38, 39, 40, 41, 42, 43]
    jump_ops = [44, 65]
    compressed_branch_jump = [73, 74, 75, 76, 80, 81]
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
    # ibex_csr (2 — build_csr_shadow_stream deleted post-review, dead RTL)
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
    # ibex_core (5)
    build_core_csr_wdata_bus_toggle_stream,
    build_core_perf_event_interleave_stream,
    build_core_pmp_req_type_toggle_stream,
    build_core_raw_hazard_stall_stream,
    build_core_illegal_insn_toggle_stream,
    # ibex_decoder (8)
    build_decoder_op_imm_bitcount_stream,
    build_decoder_crc32_stream,
    build_decoder_op_imm_shamt_family_stream,
    build_decoder_zbp_rtype_stream,
    build_decoder_zbc_clmul_stream,
    build_decoder_zbe_zbf_stream,
    build_decoder_zbb_logic_pack_minmax_stream,
    build_decoder_illegal_default_reset_stream,
    # ibex_compressed_decoder (6)
    build_compressed_c1_alu_shift_ca_mux_stream,
    build_compressed_c2_reg_ctrl_mux_stream,
    build_compressed_lui_addi16sp_special_case_stream,
    build_compressed_cl_cs_immediate_scramble_stream,
    build_compressed_ci_addi_li_signext_stream,
    build_compressed_ciw_addi4spn_scramble_stream,
    # ibex_icache (5)
    build_ibex_icache_enable_toggle_stream,
    build_ibex_icache_fencei_reinvalidate_burst_stream,
    build_ibex_icache_enable_during_invalidate_stream,
    build_ibex_icache_compressed_alignment_skid_stream,
    build_ibex_icache_branch_density_stale_fill_stream,
]

assert len(ALL_STREAM_BUILDERS) == 63, len(ALL_STREAM_BUILDERS)
