"""gen_llm_targets.py — Generează accessible_bins_for_llm.txt folosind L10 baseline.

Spre deosebire de export_for_llm.py (care folosea L9 baseline), acest script:
  1. Încarcă baseline-ul real (L10 focused, 14404 bins)
  2. Rulează clasificarea verify_ceiling pe bins-urile neacoperite
  3. Exportă doar bins-urile NEEDS + REACHABLE? grupate pe modul
  4. Scrie un nou accessible_bins_for_llm.txt compatibil cu llm_loop.py

Usage:
    python gen_llm_targets.py --hits ../level_llm/baseline_hits.pkl
    python gen_llm_targets.py --hits ../level_llm/baseline_hits.pkl --max-reachable 200
"""

import sys, re, pickle, argparse
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent
L5   = THIS.parent / "level5_real_rtl"
L7   = THIS.parent / "level7_stimulus"
sys.path.insert(0, str(L5))
sys.path.insert(0, str(L7))

import cov_parser
from analyze_unreachable import (
    TIED_OFF_SUBSTRINGS, classify, parse_point,
)

_PAGE_RE = re.compile(r"page\x02([^\x01]+)")
_SIG_RE  = re.compile(r"\x01o\x02([^\x01]+)")
_LINE_RE = re.compile(r"\x01l\x02([^\x01]+)")

def stable_id(k):
    pm = _PAGE_RE.search(k); sm = _SIG_RE.search(k); lm = _LINE_RE.search(k)
    return (f"{pm.group(1) if pm else '?'}"
            f"|{sm.group(1) if sm else '?'}"
            f"|{lm.group(1) if lm else '?'}")


MODULE_HINTS = {
    "ibex_decoder":   "Decoder signals — imm_u_type (bits 31-12, set by LUI/AUIPC), "
                      "zimm_rs1_type (5-bit CSR immediate, set by CSRRWI/CSRRSI/CSRRCI). "
                      "Use LUI imm_bucket=3 (0xFFFFF<<12) and imm_bucket=4 (0x80000<<12) "
                      "to set upper bits. Use CSRRWI/CSRRSI/CSRRCI with rs1=31 (imm=31) "
                      "to exercise zimm bits. AUIPC sets imm_u_type from PC+imm.",
    "ibex_id_stage":  "ID stage — operand mux selects and use_rs signals. "
                      "alu_op_a/b_mux_sel toggles with instruction type changes. "
                      "Mix R-type (uses rs1+rs2), I-type (uses rs1+imm), "
                      "U-type (uses PC+imm for AUIPC, zero+imm for LUI), "
                      "branches (rs1+rs2 compared, PC+imm for target). "
                      "use_rs1/use_rs2 signals toggle when instruction type changes.",
    "ibex_ex_block":  "Execute block — alu_imd_val_d[0] is the MUL/DIV intermediate "
                      "accumulator. It holds partial products during multi-cycle multiply. "
                      "Use ALL multiply variants: MUL, MULH (signed×signed high), "
                      "MULHSU (signed×unsigned high), MULHU (unsigned×unsigned high), "
                      "DIV, DIVU, REM, REMU. Use operands x17 (0x7FFFFFFF) and x18 "
                      "(0x80000000) to exercise sign-bit paths in imd_val.",
    "ibex_core":      "Core — bt_a_operand is the A input to the branch adder (PC for "
                      "JAL/branches, rs1 for JALR). bt_b_operand is the B input (imm). "
                      "To set bt_a_operand[N]: the PC must have bit N set, which happens "
                      "naturally as the program runs (PC=0x80000000+offset). "
                      "rf_wdata_fwd_wb[N]: register write data forwarded from WB — "
                      "use MUL/DIV with x17/x18/x22/x23 to produce diverse result bits. "
                      "csr_mtvec[N]: write diverse values to mtvec with CSRRW before "
                      "triggering an exception — mtvec base bits [31:2] are all writable.",
    "ibex_csr":       "CSR module — rd_data_o bits come from reading CSRs that have "
                      "non-zero content. After an ECALL: mepc=PC (has bits 31,23+ set), "
                      "mcause=11 (0xB=0b1011, sets bits 0,1,3). After EBREAK: mcause=3. "
                      "After misaligned LW: mcause=4, mtval=bad_addr. "
                      "Use CSRRS rd, mepc, x0 (rd!=x0) to read mepc into a register — "
                      "this exercises csr_rdata bits. mcycle is always incrementing.",
    "ibex_cs_registers": "CS registers — "
                         "mtvec BASE bits [31:2] ARE writable (bits [1:0] are MODE, "
                         "bit 1 is always 0). Write 0x12345678 to mtvec with CSRRW "
                         "then trigger ECALL to exercise mtvec_q[2-7] and csr_mtvec_o bits. "
                         "mcountinhibit[0]=CY (cycle inhibit) and [2]=IR (instret inhibit) "
                         "are writable; bits 3-31 are tied off (MHPMCounterNum=0). "
                         "csr_save_if_i/csr_save_id_i go high during exception entry — "
                         "trigger with ECALL. exception_pc bits come from the PC at trap.",
    "ibex_controller": "Controller FSM — transitions: BOOT→DECODE on first instr, "
                       "DECODE→FLUSH on branch taken, DECODE→WAIT_SLEEP on WFI, "
                       "DECODE→IRQ_TAKEN on interrupt (but IRQs are tied off here). "
                       "Exception path: DECODE→FLUSH, saves mepc/mcause. "
                       "Use ECALL (mcause=11), EBREAK (mcause=3), illegal instr (mcause=2), "
                       "misaligned load (mcause=4), WFI, then MRET to exercise all paths.",
    "ibex_counter":   "Counter — each counter_val_o[N] needs 2^N cycles to reach 1. "
                      "Bits 0-9 are easily reached. Bits 10-19 need 1K-512K cycles "
                      "(feasible with 256-instr programs run many times or NOPs). "
                      "Bits 20+ need >1M cycles — practically unreachable. "
                      "To maximize: fill the program with NOPs (ADDI x0,x0,0) and "
                      "use mcountinhibit=0 (default) to keep counters running.",
    "ibex_alu":       "ALU — adder_result, shift_result, comparison_result intermediate "
                      "wires. Use: ADD/SUB with x17(INT_MAX)+x18(INT_MIN) for overflow paths, "
                      "SLL/SRL/SRA with x19(0x55555555) and x21(0xAAAAAAAA) for shift paths, "
                      "SLT/SLTU with negative vs positive values for comparison paths, "
                      "XOR/OR/AND with x16(0xFFFFFFFF) for logic paths.",
    "ibex_prefetch_buffer": "Prefetch — branch_mispredict and fill_buffer signals toggle "
                             "on non-sequential PC changes. Every taken branch or JAL "
                             "causes a prefetch flush. Use: BEQ/BNE with x0 (always "
                             "not-taken or always taken), JAL to jump forward/backward. "
                             "Mix taken and not-taken branches to exercise both paths.",
    "ibex_fetch_fifo": "Fetch FIFO — rdata_outstanding and split_misaligned_access toggle "
                       "when fetch crosses a 32-bit boundary (odd halfword address). "
                       "JAL to a 2-byte aligned but not 4-byte aligned address causes "
                       "a split fetch. Use JAL imm_bucket=0 (+4 bytes) for sequential "
                       "fetches, imm_bucket=1 (+8) to skip instructions.",
    "ibex_if_stage":  "IF stage — instr_valid_id, pc_set signals. "
                      "pc_set goes high on every branch/jump taken or exception. "
                      "instr_rdata_id bits: the raw instruction word passing through IF. "
                      "To exercise diverse instruction bits: use all instruction formats "
                      "(R, I, S, B, U, J types) so the 32-bit instr word has all bit "
                      "patterns. LUI sets bits [31:12], CSRRWI sets bits [19:15] (zimm).",
    "ibex_load_store_unit": "LSU — addr_last[N] is the address of the last memory op. "
                             "addr_last comes from the ALU result for load/store. "
                             "x20=0x00010000 is the safe base; "
                             "LW x1, 100(x20) → addr=0x10064 (exercises bits 6,2). "
                             "LW x1, -2048(x20) → addr=0xF800 (exercises bits 15,11). "
                             "For misaligned (triggers exception): LW x1, 1(x20). "
                             "Use all imm_buckets {-2048,-100,0,100,2047} with x20.",
    "ibex_multdiv_fast": "Multiplier — op_numerator (dividend) and op_denominator (divisor) "
                          "are latched at start of MUL/DIV. multdiv_result[N] is the output. "
                          "Use ALL 8 variants: MUL MULH MULHSU MULHU DIV DIVU REM REMU. "
                          "Best operand pairs for bit coverage: "
                          "(x17=0x7FFFFFFF, x18=0x80000000), (x22=0x12345678, x23=0xFEDCBA98), "
                          "(x16=0xFFFFFFFF, x19=0x55555555), (x1=1, x16=-1 for edge cases).",
}

SUGGESTED_PROGRAMS = {
    "ibex_decoder": """\
[{"op":"LUI","rd":5,"rs1":0,"rs2":0,"imm_bucket":3},
 {"op":"LUI","rd":6,"rs1":0,"rs2":0,"imm_bucket":4},
 {"op":"AUIPC","rd":7,"rs1":0,"rs2":0,"imm_bucket":2},
 {"op":"CSRRWI","rd":8,"rs1":31,"rs2":0,"imm_bucket":4},
 {"op":"CSRRSI","rd":9,"rs1":15,"rs2":0,"imm_bucket":2},
 {"op":"CSRRCI","rd":10,"rs1":7,"rs2":0,"imm_bucket":1}]""",
    "ibex_ex_block": """\
[{"op":"MUL","rd":5,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"MULH","rd":6,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"MULHSU","rd":7,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"MULHU","rd":8,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"DIV","rd":9,"rs1":22,"rs2":23,"imm_bucket":0},
 {"op":"REM","rd":10,"rs1":22,"rs2":23,"imm_bucket":0}]""",
    "ibex_core": """\
[{"op":"LUI","rd":5,"rs1":0,"rs2":0,"imm_bucket":3},
 {"op":"AUIPC","rd":6,"rs1":0,"rs2":0,"imm_bucket":2},
 {"op":"MUL","rd":7,"rs1":5,"rs2":6,"imm_bucket":0},
 {"op":"ADD","rd":8,"rs1":7,"rs2":5,"imm_bucket":0},
 {"op":"BEQ","rd":0,"rs1":8,"rs2":5,"imm_bucket":1},
 {"op":"JALR","rd":1,"rs1":6,"rs2":0,"imm_bucket":0}]""",
    "ibex_cs_registers": """\
[{"op":"LUI","rd":5,"rs1":0,"rs2":0,"imm_bucket":2},
 {"op":"CSRRW","rd":6,"rs1":5,"rs2":0,"imm_bucket":1},
 {"op":"ECALL","rd":0,"rs1":0,"rs2":0,"imm_bucket":0},
 {"op":"CSRRS","rd":7,"rs1":0,"rs2":0,"imm_bucket":0},
 {"op":"ADDI","rd":8,"rs1":0,"rs2":0,"imm_bucket":4},
 {"op":"CSRRW","rd":9,"rs1":8,"rs2":0,"imm_bucket":0}]""",
    "ibex_counter": """\
[{"op":"ADDI","rd":0,"rs1":0,"rs2":0,"imm_bucket":2},
 {"op":"ADDI","rd":0,"rs1":0,"rs2":0,"imm_bucket":2},
 {"op":"ADDI","rd":0,"rs1":0,"rs2":0,"imm_bucket":2}]
# Repeat NOPs (ADDI x0,x0,0) to fill all 256 steps — counter bits increment each cycle""",
    "ibex_multdiv_fast": """\
[{"op":"MUL","rd":5,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"MULH","rd":6,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"MULHSU","rd":7,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"MULHU","rd":8,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"DIV","rd":9,"rs1":22,"rs2":1,"imm_bucket":0},
 {"op":"DIVU","rd":10,"rs1":23,"rs2":1,"imm_bucket":0},
 {"op":"REM","rd":11,"rs1":16,"rs2":22,"imm_bucket":0},
 {"op":"REMU","rd":12,"rs1":19,"rs2":21,"imm_bucket":0}]""",
    "ibex_alu": """\
[{"op":"ADD","rd":5,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"SUB","rd":6,"rs1":17,"rs2":18,"imm_bucket":0},
 {"op":"SLL","rd":7,"rs1":19,"rs2":22,"imm_bucket":0},
 {"op":"SRL","rd":8,"rs1":21,"rs2":22,"imm_bucket":0},
 {"op":"SRA","rd":9,"rs1":18,"rs2":22,"imm_bucket":0},
 {"op":"XOR","rd":10,"rs1":16,"rs2":19,"imm_bucket":0},
 {"op":"SLT","rd":11,"rs1":18,"rs2":17,"imm_bucket":0},
 {"op":"SLTU","rd":12,"rs1":16,"rs2":17,"imm_bucket":0}]""",
    "ibex_load_store_unit": """\
[{"op":"LW","rd":5,"rs1":20,"rs2":0,"imm_bucket":4},
 {"op":"LW","rd":6,"rs1":20,"rs2":0,"imm_bucket":0},
 {"op":"LW","rd":7,"rs1":20,"rs2":0,"imm_bucket":1},
 {"op":"SW","rd":0,"rs1":20,"rs2":22,"imm_bucket":3},
 {"op":"LH","rd":8,"rs1":20,"rs2":0,"imm_bucket":2},
 {"op":"LB","rd":9,"rs1":20,"rs2":0,"imm_bucket":1}]""",
    "ibex_controller": """\
[{"op":"ECALL","rd":0,"rs1":0,"rs2":0,"imm_bucket":0},
 {"op":"EBREAK","rd":0,"rs1":0,"rs2":0,"imm_bucket":0},
 {"op":"ADDI","rd":5,"rs1":0,"rs2":0,"imm_bucket":4},
 {"op":"LW","rd":6,"rs1":5,"rs2":0,"imm_bucket":0},
 {"op":"WFI","rd":0,"rs1":0,"rs2":0,"imm_bucket":0},
 {"op":"MRET","rd":0,"rs1":0,"rs2":0,"imm_bucket":0}]""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits",   required=True,
                    help="Fișier .pkl cu baseline hits (ex: ../level_llm/baseline_hits.pkl)")
    ap.add_argument("--covdat", default="../../cpu/coverage.dat")
    ap.add_argument("--max-reachable", type=int, default=150,
                    help="Max bins REACHABLE? de inclus per modul (default: 150)")
    ap.add_argument("--out", default=None,
                    help="Fișier output (default: ../level_llm/accessible_bins_for_llm.txt)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else THIS.parent / "level_llm" / "accessible_bins_for_llm.txt"

    with open(args.hits, "rb") as f:
        cum_hits = pickle.load(f)
    cum_hits_ids = {stable_id(k) for k in cum_hits}
    print(f"Baseline: {len(cum_hits_ids):,} hits")

    s = cov_parser.parse(args.covdat)
    prefix = "page\x02v_toggle/"
    all_toggle_keys = {k for k in s.points if prefix in k}
    covdat_id_to_key = {stable_id(k): k for k in all_toggle_keys}
    print(f"Coverage.dat: {len(covdat_id_to_key):,} toggle points")

    uncovered = [k for sid, k in covdat_id_to_key.items() if sid not in cum_hits_ids]
    print(f"Neacoperite: {len(uncovered):,}")

    by_module = defaultdict(lambda: {"NEEDS": [], "REACHABLE?": []})
    for key in uncovered:
        page, sig, line = parse_point(key)
        tag = classify(sig, key)
        if tag.startswith("TIED"):
            continue
        bucket = "NEEDS" if tag.startswith("NEEDS") else "REACHABLE?"
        module = page[len("v_toggle/"):].split("__")[0] if page.startswith("v_toggle/") else page
        by_module[module][bucket].append((sig, line))

    total_bins = sum(len(v["NEEDS"]) + len(v["REACHABLE?"]) for v in by_module.values())
    print(f"Bins NEEDS + REACHABLE?: {total_bins:,}")

    # Sortează modulele: NEEDS first, apoi REACHABLE? cu mai multe bins
    def module_score(item):
        m, d = item
        return -(len(d["NEEDS"]) * 10 + len(d["REACHABLE?"]))
    sorted_modules = sorted(by_module.items(), key=module_score)

    lines = []
    lines.append("=" * 70)
    lines.append(f"ACCESSIBLE BINS FOR LLM — genererat din L10 baseline ({len(cum_hits_ids):,} hits)")
    lines.append(f"Total bins: {total_bins}")
    lines.append("=" * 70)
    lines.append("")

    group_id = 1
    for module, data in sorted_modules:
        needs = data["NEEDS"]
        reachable = data["REACHABLE?"][:args.max_reachable]
        all_bins = needs + reachable
        if not all_bins:
            continue

        hint = MODULE_HINTS.get(module, f"Signals in {module} — use diverse instruction sequences.")
        suggested = SUGGESTED_PROGRAMS.get(module, "(see hint above)")

        lines.append("=" * 70)
        lines.append(f"GROUP {group_id}: {module} ({len(all_bins)} bins)")
        lines.append("")
        lines.append("EXPLANATION:")
        if needs:
            lines.append(f"  {len(needs)} NEEDS bins (known stimulus exists) + "
                         f"{len(reachable)} REACHABLE? bins.")
        lines.append(f"  {hint}")
        lines.append("")
        lines.append("SIGNALS:")
        for sig, line in all_bins:
            lines.append(f"  line_{line:<6s}  {sig}")
        lines.append("")
        lines.append("SUGGESTED PROGRAM:")
        lines.append(suggested)
        lines.append("")
        group_id += 1

    out_path.write_text("\n".join(lines))
    print(f"\nExportat {group_id-1} grupuri → {out_path}")
    print(f"Rulează: python llm_loop.py --provider ... (folosește noul fișier automat)")


if __name__ == "__main__":
    main()
