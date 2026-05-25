"""starter.py — Entry point for LLM-based coverage starting from L10 baseline.

This script:
  1. Loads the L10 baseline hits (14,404 bins = 71.99% coverage)
  2. Provides run_and_check() — run a program and see which NEW bins it covers
  3. Shows a working example for each group in accessible_bins_for_llm.txt

Usage:
    python starter.py                    # run built-in example programs
    python starter.py --group mtvec      # run only one group
    python starter.py --list             # list all accessible bins
"""

import sys, re, pickle, argparse, time
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent
L5   = THIS.parent / "level5_real_rtl"
L7   = THIS.parent / "level7_stimulus"
L9   = THIS.parent / "level9_ops"
sys.path.insert(0, str(L5))
sys.path.insert(0, str(L9))

import cov_parser
from env_l9_v2 import run_program

# ── Op constants (from codec_l9) ─────────────────────────────────────────────
ADD=0;  SUB=1;  SLL=2;  SLT=3;  SLTU=4; XOR=5;  SRL=6;  SRA=7;  OR=8;  AND=9
ADDI=10; SLTI=11; SLTIU=12; XORI=13; ORI=14; ANDI=15
SLLI=16; SRLI=17; SRAI=18
LB=19; LH=20; LW=21; LBU=22; LHU=23
SB=24; SH=25; SW=26
CSRRW=27; CSRRS=28; CSRRC=29
MUL=30; MULH=31; MULHSU=32; MULHU=33
DIV=34; DIVU=35; REM=36; REMU=37
BEQ=38; BNE=39; BLT=40; BGE=41; BLTU=42; BGEU=43
JAL=44; AUIPC=61; ECALL=62; EBREAK=63; LUI=64; JALR=65
MRET=70; WFI=71; FENCE_I=72
NOP = (ADDI, 0, 0, 0, 0)

# ── Stable ID (for matching pkl hits with coverage.dat) ──────────────────────
_PAGE_RE = re.compile(r"page\x02([^\x01]+)")
_SIG_RE  = re.compile(r"\x01o\x02([^\x01]+)")
_LINE_RE = re.compile(r"\x01l\x02([^\x01]+)")

def stable_id(k):
    pm = _PAGE_RE.search(k); sm = _SIG_RE.search(k); lm = _LINE_RE.search(k)
    return (f"{pm.group(1) if pm else '?'}"
            f"|{sm.group(1) if sm else '?'}"
            f"|{lm.group(1) if lm else '?'}")


# ── Load baseline ─────────────────────────────────────────────────────────────
BASELINE_PKL = THIS / "baseline_hits.pkl"
COVDAT       = THIS.parent.parent / "cpu" / "coverage.dat"

def load_baseline():
    with open(BASELINE_PKL, "rb") as f:
        hits = pickle.load(f)
    print(f"Baseline loaded: {len(hits):,} hits (71.99%)")
    return hits

def load_accessible_bins():
    """Parse accessible_bins_for_llm.txt → set of signal names."""
    bins = set()
    with open(THIS / "accessible_bins_for_llm.txt") as f:
        for line in f:
            m = re.match(r"\s{2}(\S+)\s{2,}(\S+)", line)
            if m:
                bins.add(m.group(2))
    return bins


# ── Core function: run a program and check new hits ───────────────────────────
def run_and_check(actions: list, baseline_ids: set, covdat_map: dict,
                  target_bins: set = None) -> dict:
    """
    Run a program and return newly covered bins.

    Args:
        actions:      list of (op, rd, rs1, rs2, imm_bucket) tuples
        baseline_ids: set of stable_ids already covered (from baseline pkl)
        covdat_map:   dict stable_id → signal_name (from coverage.dat)
        target_bins:  optional set of signal names to focus on

    Returns:
        dict with:
          'new_total':  count of new bins vs baseline
          'new_target': new bins that are in the accessible list
          'signals':    list of newly covered signal names
    """
    summary = run_program(actions)
    if summary is None:
        return {'new_total': 0, 'new_target': 0, 'signals': []}

    # Get hits from this run
    run_hits_ids = {stable_id(k)
                    for k, v in summary.points.items()
                    if v > 0 and "v_toggle/" in k}

    # New = in this run but NOT in baseline
    new_ids = run_hits_ids - baseline_ids

    # Map back to signal names
    new_signals = []
    for sid in new_ids:
        if sid in covdat_map:
            new_signals.append(covdat_map[sid])

    new_target = [s for s in new_signals if s in target_bins] if target_bins else new_signals

    return {
        'new_total':  len(new_ids),
        'new_target': len(new_target),
        'signals':    sorted(new_signals),
        'target_signals': sorted(new_target),
    }


# ── Example programs for each group ──────────────────────────────────────────

def prog_mcountinhibit(n=50):
    """Group 1: mcountinhibit CSR bits [1:31] — write various patterns."""
    acts = []
    # CSR bucket for mcountinhibit (0x320) — bucket index in codec_l9
    # From codec_l9: CSRS list, mcountinhibit is at index ~22
    CSR_MCOUNTINHIBIT = 22  # adjust if needed
    for _ in range(n // 10):
        for imm_b in range(5):   # various imm buckets → different LUI values
            acts += [
                (LUI,   1, 0, 0, imm_b),     # lui t1, <value>
                (CSRRW, 0, 1, 0, CSR_MCOUNTINHIBIT),  # csrw mcountinhibit, t1
                (ADDI,  1, 0, 0, 0),          # li t1, 0
                (CSRRW, 0, 1, 0, CSR_MCOUNTINHIBIT),  # csrw mcountinhibit, 0
            ]
    return acts


def prog_mtvec_bits(n=50):
    """Group 2: mtvec bits [7:1] — write 0xFE then 0x00."""
    acts = []
    CSR_MTVEC = 5  # mtvec bucket index in codec_l9
    for _ in range(n // 5):
        for imm_b in range(5):
            acts += [
                (LUI,   1, 0, 0, imm_b),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
                (ADDI,  1, 0, 0, 0),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
                (ADDI,  1, 0, 0, 1),          # imm=1 → bits [11:0]
                (CSRRW, 0, 1, 0, CSR_MTVEC),
                (ADDI,  1, 0, 0, 0),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
            ]
    return acts


def prog_exception_paths(n=60):
    """Group 3: exception paths — illegal + misaligned."""
    acts = []
    for _ in range(n // 6):
        acts += [
            (ECALL,  0, 0, 0, 0),    # environment call → exception
            (EBREAK, 0, 0, 0, 0),    # breakpoint → exception
            NOP, NOP,
            (MRET,   0, 0, 0, 0),    # return from exception
            NOP,
        ]
    return acts


def prog_large_immediates(n=60):
    """Group 4: immediate upper bits — LUI/AUIPC with varied large values."""
    acts = []
    imm_vals = list(range(5))  # all imm buckets
    for _ in range(n // (len(imm_vals) * 2)):
        for ib in imm_vals:
            acts += [
                (LUI,   1, 0, 0, ib),
                (AUIPC, 2, 0, 0, ib),
                (LUI,   3, 0, 0, (ib + 2) % 5),
                (AUIPC, 4, 0, 0, (ib + 3) % 5),
            ]
    return acts


def prog_csr_read_data(n=50):
    """Group 5: CSR read data bits — read various CSRs with non-zero values."""
    acts = []
    # Write to mtvec then read back repeatedly
    CSR_MTVEC    = 5
    CSR_MSTATUS  = 0
    CSR_MIE      = 4
    for _ in range(n // 8):
        for ib in range(5):
            acts += [
                (LUI,   1, 0, 0, ib),
                (CSRRW, 2, 1, 0, CSR_MTVEC),    # write + read mtvec
                (CSRRS, 3, 0, 0, CSR_MSTATUS),  # read mstatus
                (CSRRS, 4, 0, 0, CSR_MIE),       # read mie
                NOP, NOP,
                (ADDI,  1, 0, 0, 0),
                (CSRRW, 0, 1, 0, CSR_MTVEC),
            ]
    return acts


def prog_alu_decoder(n=60):
    """Group 7: ALU/decoder — MUL/DIV, JAL/JALR, ctrl FSM."""
    acts = []
    for _ in range(n // 6):
        for rs1 in range(1, 5):
            for rs2 in range(1, 5):
                acts += [
                    (MUL,  1, rs1, rs2, 0),
                    (DIV,  2, rs1, rs2, 0),
                    (MULH, 3, rs1, rs2, 0),
                ]
                if len(acts) >= n:
                    break
    return acts[:n]


def prog_data_alignment(n=40):
    """Group 6: data address alignment bits — LB/SB to odd/2-byte addresses."""
    acts = []
    for _ in range(n // 4):
        acts += [
            (LB,  1, 0, 0, 1),   # lb t1, 1(x0) → odd address → bit0=1
            (SB,  1, 0, 0, 1),   # sb t1, 1(x0)
            (LH,  1, 0, 0, 2),   # lh t1, 2(x0) → 2-byte → bit1=1
            (SH,  1, 0, 0, 2),   # sh t1, 2(x0)
        ]
    return acts


# ── Main ──────────────────────────────────────────────────────────────────────

PROGRAMS = {
    "mcountinhibit": (prog_mcountinhibit, "mcountinhibit CSR bits"),
    "mtvec":         (prog_mtvec_bits,    "mtvec low bits"),
    "exception":     (prog_exception_paths, "exception paths"),
    "immediates":    (prog_large_immediates, "large immediate bits"),
    "csr_read":      (prog_csr_read_data,   "CSR read data"),
    "alu":           (prog_alu_decoder,     "ALU/decoder signals"),
    "alignment":     (prog_data_alignment,  "data address alignment"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group",  default=None, help="Run only one group")
    ap.add_argument("--list",   action="store_true", help="List accessible bins")
    ap.add_argument("--covdat", default=str(COVDAT))
    args = ap.parse_args()

    if args.list:
        bins = load_accessible_bins()
        print(f"Accessible bins ({len(bins)}):")
        for b in sorted(bins):
            print(f"  {b}")
        return

    # ── Setup ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("LLM Coverage — starting from L10 baseline (71.99%)")
    print("=" * 60)

    baseline_hits = load_baseline()
    baseline_ids  = {stable_id(k) for k in baseline_hits}

    s = cov_parser.parse(args.covdat)
    prefix = "page\x02v_toggle/"
    covdat_map = {}
    for k in s.points:
        if prefix in k:
            sid  = stable_id(k)
            # extract signal name
            sm = _SIG_RE.search(k)
            if sm:
                covdat_map[sid] = sm.group(1)

    target_bins = load_accessible_bins()
    print(f"Target bins: {len(target_bins)}")
    print()

    # ── Run programs ──────────────────────────────────────────────────────
    groups_to_run = ([args.group] if args.group else list(PROGRAMS.keys()))

    total_new_target = 0
    cumulative_covered = set()

    for name in groups_to_run:
        if name not in PROGRAMS:
            print(f"Unknown group: {name}. Available: {list(PROGRAMS.keys())}")
            continue

        prog_fn, desc = PROGRAMS[name]
        print(f"▶ Group: {desc}")
        t0 = time.time()

        actions = prog_fn()
        result  = run_and_check(actions, baseline_ids | cumulative_covered,
                                 covdat_map, target_bins)

        elapsed = time.time() - t0
        print(f"  New bins (total):  {result['new_total']}")
        print(f"  New bins (target): {result['new_target']}  ← from accessible list")
        if result['target_signals']:
            for sig in result['target_signals']:
                print(f"    ✅ {sig}")
                cumulative_covered.add(sig)
        else:
            print(f"    (none from target list)")
        print(f"  Time: {elapsed:.1f}s")
        print()
        total_new_target += result['new_target']

    print("=" * 60)
    covered_now = 14404 + total_new_target
    pct = 100 * covered_now / 20023
    print(f"RESULT: {total_new_target} new bins covered from target list")
    print(f"Coverage: {covered_now:,} / 20,023 = {pct:.2f}%")
    print(f"  (was 71.99% → now {pct:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
