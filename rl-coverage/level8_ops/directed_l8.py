"""directed_l8.py — Stimul dirijat spre semnalele REACHABLE? neacoperite la L8.

Atacă 4 categorii identificate din analiza de ceiling:
  1. Misaligned accesses  → lsu_err, ls_fsm_cs, mcause={4,6}, mtval bits
  2. Counter high writes  → counterh_we_i (CSRRW la mcycleh/minstreth)
  3. MUL/DIV diverse      → multdiv_operand_a/b bits, div_by_zero_q
  4. CSR bit walking      → csr_rdata/wr_data bits

Generează machine code raw (bypass codec) și rulează prin aceeași
infrastructură Vtop ca env_l8.py.

Usage:
    python directed_l8.py
"""

import json, os, subprocess, sys, time
from pathlib import Path

THIS  = Path(__file__).resolve().parent
L5    = (THIS.parent / "level5_real_rtl").resolve()
ML4DV = (THIS.parent.parent / "cpu").resolve()
VTOP  = ML4DV / "sim_build" / "Vtop"
COVDAT = ML4DV / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l8_directed.json"

sys.path.insert(0, str(L5))
import cov_parser

# ── CSR addresses ──────────────────────────────────────────────────────────
CSR_MCYCLE       = 0xB00
CSR_MCYCLEH      = 0xB80
CSR_MINSTRET     = 0xB02
CSR_MINSTRETH    = 0xB82
CSR_MSCRATCH     = 0x340
CSR_MEPC         = 0x341
CSR_MCAUSE       = 0x342
CSR_MTVAL        = 0x343
CSR_MCOUNTINHIBIT= 0x320

# ── Instruction encoders ───────────────────────────────────────────────────

def _imm12(v):
    return int(v) & 0xFFF

def addi(rd, rs1, imm):
    return (_imm12(imm) << 20) | ((rs1 & 31) << 15) | (0b000 << 12) | ((rd & 31) << 7) | 0b0010011

def lui(rd, imm20):
    return ((int(imm20) & 0xFFFFF) << 12) | ((rd & 31) << 7) | 0b0110111

def lw(rd, rs1, offset=0):
    return (_imm12(offset) << 20) | ((rs1 & 31) << 15) | (0b010 << 12) | ((rd & 31) << 7) | 0b0000011

def lh(rd, rs1, offset=0):
    return (_imm12(offset) << 20) | ((rs1 & 31) << 15) | (0b001 << 12) | ((rd & 31) << 7) | 0b0000011

def lhu(rd, rs1, offset=0):
    return (_imm12(offset) << 20) | ((rs1 & 31) << 15) | (0b101 << 12) | ((rd & 31) << 7) | 0b0000011

def lb(rd, rs1, offset=0):
    return (_imm12(offset) << 20) | ((rs1 & 31) << 15) | (0b000 << 12) | ((rd & 31) << 7) | 0b0000011

def sw(rs2, rs1, offset=0):
    o = _imm12(offset)
    return ((o >> 5) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b010 << 12) | ((o & 0x1F) << 7) | 0b0100011

def sh(rs2, rs1, offset=0):
    o = _imm12(offset)
    return ((o >> 5) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b001 << 12) | ((o & 0x1F) << 7) | 0b0100011

def sb(rs2, rs1, offset=0):
    o = _imm12(offset)
    return ((o >> 5) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b000 << 12) | ((o & 0x1F) << 7) | 0b0100011

def csrrw(rd, csr, rs1):
    return ((csr & 0xFFF) << 20) | ((rs1 & 31) << 15) | (0b001 << 12) | ((rd & 31) << 7) | 0b1110011

def csrrs(rd, csr, rs1):
    return ((csr & 0xFFF) << 20) | ((rs1 & 31) << 15) | (0b010 << 12) | ((rd & 31) << 7) | 0b1110011

def csrrc(rd, csr, rs1):
    return ((csr & 0xFFF) << 20) | ((rs1 & 31) << 15) | (0b011 << 12) | ((rd & 31) << 7) | 0b1110011

def mul(rd, rs1, rs2):
    return (0b0000001 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b000 << 12) | ((rd & 31) << 7) | 0b0110011

def mulh(rd, rs1, rs2):
    return (0b0000001 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b001 << 12) | ((rd & 31) << 7) | 0b0110011

def mulhsu(rd, rs1, rs2):
    return (0b0000001 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b010 << 12) | ((rd & 31) << 7) | 0b0110011

def mulhu(rd, rs1, rs2):
    return (0b0000001 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b011 << 12) | ((rd & 31) << 7) | 0b0110011

def div(rd, rs1, rs2):
    return (0b0000001 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b100 << 12) | ((rd & 31) << 7) | 0b0110011

def divu(rd, rs1, rs2):
    return (0b0000001 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b101 << 12) | ((rd & 31) << 7) | 0b0110011

def rem(rd, rs1, rs2):
    return (0b0000001 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b110 << 12) | ((rd & 31) << 7) | 0b0110011

def remu(rd, rs1, rs2):
    return (0b0000001 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b111 << 12) | ((rd & 31) << 7) | 0b0110011

def add(rd, rs1, rs2):
    return (0b0000000 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | (0b000 << 12) | ((rd & 31) << 7) | 0b0110011

def nop():
    return addi(0, 0, 0)

# ── Programe diriguite ────────────────────────────────────────────────────

def prog_misaligned():
    """Accese misaligned → lsu_err, ls_fsm_cs, mcause={4,6}, mtval bits diverse."""
    p = []
    # Prologue registre diverse deja setate de test_run_for_l8:
    # x16=-1, x17=INT_MAX, x18=INT_MIN, x19=0x55555555, x20=0x00010000
    # x21=0xAAAAAAAA, x22=0x12345678, x23=0xFEDCBA98

    # Adrese misaligned: 1, 2, 3, 5, 7, 9, 11
    for offset in [1, 2, 3, 5, 7, 9, 11, 13]:
        p.append(addi(5, 0, offset))      # x5 = offset (misaligned addr)
        p.append(lw(6,  5, 0))            # misaligned LW  → mcause=4
        p.append(lh(7,  5, 0))            # misaligned LH  → mcause=4 (dacă odd)
        p.append(sw(16, 5, 0))            # misaligned SW  → mcause=6
        p.append(sh(17, 5, 0))            # misaligned SH  → mcause=6 (dacă odd)

    # Adrese cu biți înalți (pentru mtval bits înalți)
    for val in [0x101, 0x201, 0x401, 0x801, 0x1001]:
        p.append(addi(5, 0, val & 0x7FF))  # limitat la 12-bit signed
        p.append(lw(6, 5, 0))
        p.append(sw(19, 5, 0))

    # Combinații cu registrele diverse din prologue (x20 = 0x10000, adaugăm offset impar)
    p.append(addi(5, 20, 1))   # x5 = 0x10001 (misaligned)
    p.append(lw(6, 5, 0))
    p.append(lh(7, 5, 0))
    p.append(sw(21, 5, 0))

    p.append(addi(5, 20, 3))   # x5 = 0x10003
    p.append(lw(6, 5, 0))
    p.append(sw(22, 5, 0))

    # Accese LHU, LB (pentru diversitate)
    p.append(addi(5, 0, 1))
    p.append(lhu(6, 5, 0))
    p.append(lb(7, 5, 0))

    return p


def prog_counter_writes():
    """Scrieri în counterh → counterh_we_i, counter_val_o biți înalți."""
    p = []
    # x16=-1, x17=INT_MAX, x18=INT_MIN, x19=0x55555555, x21=0xAAAAAAAA

    for csr in [CSR_MCYCLEH, CSR_MINSTRETH]:
        for rs in [16, 17, 18, 19, 21, 22, 23]:
            p.append(csrrw(1, csr, rs))   # scrie valoare diversă în counterh
            p.append(csrrs(2, csr, rs))   # set bits
            p.append(csrrc(3, csr, rs))   # clear bits

    # Scrieri și în lower halves pentru diversitate completă
    for csr in [CSR_MCYCLE, CSR_MINSTRET]:
        for rs in [16, 17, 18, 19]:
            p.append(csrrw(1, csr, rs))

    # Mcountinhibit — dezactivează/activează contoarele (toggle we)
    p.append(csrrw(1, CSR_MCOUNTINHIBIT, 16))   # disable all counters
    p.append(csrrw(1, CSR_MCOUNTINHIBIT, 0))    # re-enable (scrie 0)

    # Walking bits pentru a togla biți individuali
    bit = 1
    for _ in range(20):
        p.append(addi(5, 0, bit & 0x7FF))
        p.append(csrrw(0, CSR_MCYCLEH, 5))
        p.append(csrrw(0, CSR_MINSTRETH, 5))
        bit = (bit << 1) & 0xFFFFF  # maxim 20 biți

    return p


def prog_multdiv_diverse():
    """MUL/DIV cu operanzi diversi → multdiv_operand_a/b biți, div_by_zero_q."""
    p = []
    # x16=-1, x17=INT_MAX, x18=INT_MIN, x19=0x55555555
    # x21=0xAAAAAAAA, x22=0x12345678, x23=0xFEDCBA98

    regs = [16, 17, 18, 19, 21, 22, 23]

    # Toate combinațiile de MUL cu registrele diverse
    for a in regs:
        for b in regs:
            p.append(mul(1, a, b))

    # MULH, MULHSU, MULHU cu valori diverse
    for a in regs:
        for b in [16, 17, 18, 19]:
            p.append(mulh(2, a, b))
            p.append(mulhsu(3, a, b))
            p.append(mulhu(4, a, b))

    # DIV/REM cu valori diverse
    for a in regs:
        for b in regs:
            p.append(div(5, a, b))
            p.append(divu(6, a, b))
            p.append(rem(7, a, b))
            p.append(remu(8, a, b))

    # DIV by zero explicit → div_by_zero_q
    for a in regs:
        p.append(div(1,  a, 0))    # DIV  a / x0 (x0=0 always)
        p.append(divu(2, a, 0))    # DIVU a / x0
        p.append(rem(3,  a, 0))    # REM  a % x0
        p.append(remu(4, a, 0))    # REMU a % x0

    # DIV INT_MIN / -1 (overflow case)
    p.append(div(1, 18, 16))       # INT_MIN / -1

    return p


def prog_csr_bitwalking():
    """Walking bits în CSR-uri writeable → csr_rdata/wr_data biți."""
    p = []

    # mscratch — complet writeable, safe
    bit = 1
    for _ in range(32):
        p.append(addi(5, 0, bit & 0x7FF) if bit < 0x800 else addi(5, 0, -1))
        p.append(csrrw(1, CSR_MSCRATCH, 5))
        p.append(csrrs(2, CSR_MSCRATCH, 5))
        p.append(csrrc(3, CSR_MSCRATCH, 5))
        bit = (bit << 1) if bit < (1 << 31) else 1

    # Valori speciale în mscratch
    for rs in [16, 17, 18, 19, 21, 22, 23]:
        p.append(csrrw(1, CSR_MSCRATCH, rs))
        p.append(csrrs(2, CSR_MSCRATCH, rs))
        p.append(csrrc(3, CSR_MSCRATCH, rs))

    # Citiri din mcause/mtval/mepc după excepții (forțăm ECALL)
    for _ in range(5):
        p.append(addi(0, 0, 0))       # NOP — lasă excepțiile să se acumuleze
        p.append(csrrs(4, CSR_MCAUSE, 0))   # citește mcause
        p.append(csrrs(5, CSR_MTVAL,  0))   # citește mtval
        p.append(csrrs(6, CSR_MEPC,   0))   # citește mepc

    return p


# ── Runner ────────────────────────────────────────────────────────────────

def run_raw(machine_code):
    """Rulează o listă de 32-bit words prin Vtop și returnează summary."""
    payload = {
        "n": len(machine_code),
        "agent": "l8",
        "machine_code": [int(w) for w in machine_code],
    }
    with open(PROGRAM_JSON, "w") as f:
        json.dump(payload, f)

    env = os.environ.copy()
    cocotb_libs = "/home/andrei/ibex_env/lib/python3.12/site-packages/cocotb/libs"
    env["LD_LIBRARY_PATH"] = (
        cocotb_libs + ":/usr/lib/x86_64-linux-gnu"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = (
        str(ML4DV) + ":/home/andrei/ibex_env/lib/python3.12/site-packages"
        + ":" + env.get("PYTHONPATH", "")
    )
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = PROGRAM_JSON

    proc = subprocess.run(
        [str(VTOP)], cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


def coverage_pct(summary, kind="toggle"):
    covered, total = summary.by_kind[kind]
    return 100.0 * covered / total if total else 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def cum_hits_from(summary):
    """Returnează setul de toggle points acoperite din summary."""
    prefix = "\x01page\x02v_toggle/"
    return {k for k, v in summary.points.items()
            if v > 0 and prefix in ("\x01" + k)}


def main():
    import numpy as np

    print("=" * 60)
    print("L8 Directed Stimulus — atac semnale REACHABLE?")
    print("=" * 60)

    programs = [
        ("Misaligned accesses (lsu_err, mcause 4/6, mtval)", prog_misaligned()),
        ("Counter high writes (counterh_we_i)",               prog_counter_writes()),
        ("MUL/DIV diverse + div_by_zero",                     prog_multdiv_diverse()),
        ("CSR bit walking (wr_data bits)",                    prog_csr_bitwalking()),
    ]

    # ── Pas 1: baseline random (5 episoade) ───────────────────────────────
    print("\nPas 1: Baseline random (5 ep × 1024 pași)...")
    from env_l8 import IbexL8Env
    from codec_l8 import N_OPS, IMM_BUCKETS

    env = IbexL8Env(episode_steps=1024, seed=99)
    rng = np.random.default_rng(99)
    cum_hits = set()
    total_toggle = None

    for ep in range(5):
        env.reset()
        t0 = time.time()
        for _ in range(1024):
            a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
                 rng.integers(32), rng.integers(IMM_BUCKETS)]
            _, _, _, trunc, info = env.step(a)
            if trunc:
                break
        dt = time.time() - t0
        # Ia coverage.dat după acest episod
        s = cov_parser.parse(str(COVDAT))
        hits = cum_hits_from(s)
        cum_hits |= hits
        if total_toggle is None:
            total_toggle = s.by_kind["toggle"][1]
        cum_pct = 100.0 * len(cum_hits) / total_toggle
        print(f"  ep {ep+1}: ep={info['ep_pct']:.2f}%  cum={cum_pct:.2f}%  ({dt:.0f}s)")

    baseline_cum = 100.0 * len(cum_hits) / total_toggle
    print(f"\n  Baseline cumulativ după 5 ep random: {baseline_cum:.2f}%")

    # ── Pas 2: adaugă programele diriguite ────────────────────────────────
    print("\nPas 2: Programe diriguite (pe lângă baseline)...")
    print(f"{'Program':<45s} {'ep%':>6s} {'new hits':>9s} {'cum%':>7s}")
    print("-" * 72)

    for name, prog in programs:
        t0 = time.time()
        s = run_raw(prog)
        dt = time.time() - t0

        if s is None:
            print(f"  [FAIL] {name}")
            continue

        hits = cum_hits_from(s)
        new_hits = hits - cum_hits
        cum_hits |= hits
        ep_pct  = 100.0 * len(hits) / total_toggle
        cum_pct = 100.0 * len(cum_hits) / total_toggle
        print(f"  {name:<43s} {ep_pct:>5.2f}%  +{len(new_hits):>6}  {cum_pct:>6.2f}%  ({dt:.0f}s)")

    final_cum = 100.0 * len(cum_hits) / total_toggle
    print(f"\n{'='*60}")
    print(f"  Baseline random (5 ep):        {baseline_cum:.2f}%")
    print(f"  După directed stimulus:        {final_cum:.2f}%")
    print(f"  Câștig net directed:          {final_cum - baseline_cum:+.2f} pp")
    print(f"  Ceiling L8:                   ~74.72%")
    print(f"  Distanță față de ceiling:     {74.72 - final_cum:.2f} pp")


if __name__ == "__main__":
    main()
