"""
directed_l8_targeted.py — Program țintit pe liniile neacoperite identificate.

Rulează DUPĂ directed_l8.py (care acoperă bazele).
Acoperă specific:
  1. Accese misaligned (SW/SH/LW/LH la adrese nealiniate)
  2. Compressed branches: C.BEQZ, C.BNEZ, C.J, C.JAL, C.LWSP
  3. FENCE.I (pentru icache_inval_o în decoder)
  4. JALR cu diverse funct3 (path-uri ilegale → illegal_insn)
  5. Instrucțiuni ilegale compressed (illegal_instr_o paths)
"""

import json, os, sys, subprocess
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
import cov_parser

ML4DV  = (THIS.parent.parent / "cpu").resolve()
_SIM_BUILD = os.environ.get("IBEX_SIM_BUILD", "sim_build")
VTOP   = ML4DV / _SIM_BUILD / "Vtop"
COVDAT = ML4DV / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l8_targeted.json"

# ── Encodere ─────────────────────────────────────────────────────────────
def addi(rd,rs1,imm): return ((imm&0xFFF)<<20)|((rs1&0x1F)<<15)|(0<<12)|((rd&0x1F)<<7)|0b0010011
def lui(rd,i20): return ((i20&0xFFFFF)<<12)|((rd&0x1F)<<7)|0b0110111
def load(f3,rd,rs1,imm): return ((imm&0xFFF)<<20)|((rs1&0x1F)<<15)|(f3<<12)|((rd&0x1F)<<7)|0b0000011
def store(f3,rs2,rs1,imm):
    return (((imm>>5)&0x7F)<<25)|((rs2&0x1F)<<20)|((rs1&0x1F)<<15)|(f3<<12)|((imm&0x1F)<<7)|0b0100011
def jalr(rd,rs1,off): return ((off&0xFFF)<<20)|((rs1&0x1F)<<15)|(0<<12)|((rd&0x1F)<<7)|0b1100111
def jal(rd,off):
    o=off&0x1FFFFF
    return (((o>>20)&1)<<31)|(((o>>1)&0x3FF)<<21)|(((o>>11)&1)<<20)|(((o>>12)&0xFF)<<12)|((rd&0x1F)<<7)|0b1101111
def nop(): return addi(0,0,0)
def fence_i(): return 0x0000100F   # FENCE.I — declanșează icache_inval_o

# ── Compressed instruction encoders ──────────────────────────────────────
def pack_c(hw):
    """Împachetează o instrucțiune compressed de 16 biți în cuvânt de 32 biți."""
    # NOP (ADDI x0,x0,0) în upper 16 biți + instrucțiunea C în lower 16 biți
    return (0x0001 << 16) | (hw & 0xFFFF)

def c_j(offset):
    """C.J — jump necondiționat, quadrant 01, funct3=101.
    Offset: 11 biți, multiplu de 2.
    Encoding: [15:13]=101 [12]=off[11] [11:9]=off[4,9,8]
               [8]=off[10] [7]=off[6] [6]=off[7] [5:4]=off[3:2]
               [3:2]=off[5] [1:0]=01  (standard CJ format)
    """
    o = offset & 0x7FE   # 11-bit signed, bit 0 = 0
    imm = (((o>>11)&1)<<12)|(((o>>4)&1)<<11)|(((o>>8)&3)<<9)|\
          (((o>>10)&1)<<8)|(((o>>6)&1)<<7)|(((o>>7)&1)<<6)|\
          (((o>>1)&0xF)<<3)|(((o>>5)&1)<<2)
    return pack_c(0b101_0000000000_01 | (imm & 0x1FFE))

def c_jal(offset):
    """C.JAL — jump and link x1, RV32 only, quadrant 01, funct3=001."""
    o = offset & 0x7FE
    imm = (((o>>11)&1)<<12)|(((o>>4)&1)<<11)|(((o>>8)&3)<<9)|\
          (((o>>10)&1)<<8)|(((o>>6)&1)<<7)|(((o>>7)&1)<<6)|\
          (((o>>1)&0xF)<<3)|(((o>>5)&1)<<2)
    return pack_c(0b001_0000000000_01 | (imm & 0x1FFE))

def c_beqz(rs1_prime, offset):
    """C.BEQZ — branch if rs1'==0, quadrant 01, funct3=110.
    rs1_prime: registru 0-7 (mapat la x8-x15).
    offset: 8 biți, multiplu de 2.
    """
    rs1p = rs1_prime & 0x7  # 3 biți
    o = offset & 0x1FE       # 8 biți signed
    # [15:13]=110 [12]=off[8] [11:10]=off[4:3] [9:7]=rs1'
    # [6:5]=off[7:6] [4:3]=off[2:1] [2]=off[5] [1:0]=01
    imm = (((o>>8)&1)<<12)|(((o>>3)&3)<<10)|(rs1p<<7)|\
          (((o>>6)&3)<<5)|(((o>>1)&3)<<3)|(((o>>5)&1)<<2)
    return pack_c(0b110_00_000_00_00_0_01 | (imm & 0x1C7C) | (rs1p << 7))

def c_bnez(rs1_prime, offset):
    """C.BNEZ — branch if rs1'!=0, quadrant 01, funct3=111."""
    rs1p = rs1_prime & 0x7
    o = offset & 0x1FE
    imm = (((o>>8)&1)<<12)|(((o>>3)&3)<<10)|(rs1p<<7)|\
          (((o>>6)&3)<<5)|(((o>>1)&3)<<3)|(((o>>5)&1)<<2)
    return pack_c(0b111_00_000_00_00_0_01 | (imm & 0x1C7C) | (rs1p << 7))

def c_lwsp(rd, offset):
    """C.LWSP — load word stack pointer, quadrant 10, funct3=010.
    rd != 0, offset: 6 biți (word-aligned).
    """
    rd = rd & 0x1F
    o = (offset >> 2) & 0xF   # bits [5:2]
    o2 = (offset >> 6) & 0x3  # bits [7:6]
    # [15:13]=010 [12]=off[5] [11:7]=rd [6:4]=off[4:2] [3:2]=off[7:6] [1:0]=10
    hw = (0b010 << 13) | (((offset>>5)&1) << 12) | (rd << 7) | \
         (((offset>>2)&7) << 4) | (((offset>>6)&3) << 2) | 0b10
    return pack_c(hw)

def c_li(rd, imm6):
    """C.LI — load immediate, quadrant 01, funct3=010."""
    nz = imm6 & 0x3F
    hw = (0b010 << 13) | (((nz>>5)&1) << 12) | ((rd&0x1F) << 7) | \
         ((nz&0x1F) << 2) | 0b01
    return pack_c(hw)

def c_nop():
    return pack_c(0x0001)  # C.NOP

def build_targeted_program():
    p = []
    E = p.append

    # ── Setup regiștri de bază ────────────────────────────────────────────
    E(lui(20, 0x00010))          # x20 = 0x10000 (baza memoriei)
    for i in range(1, 12):
        E(addi(i, 0, i))         # x1..x11 = 1..11

    # ── 1. FENCE.I (icache_inval_o în decoder) ────────────────────────────
    for _ in range(3):
        E(fence_i())
        E(nop())

    # ── 2. Accese MISALIGNED ──────────────────────────────────────────────
    # Misaligned WORD store/load (adresă +1, +2, +3 față de word-aligned)
    for mis in [1, 2, 3]:
        E(addi(26, 20, mis))         # x26 = baza + mis (misaligned)
        E(store(0b010, 1, 26, 0))    # SW x1, 0(x26) — misaligned store word
        E(load(0b010, 25, 26, 0))    # LW x25, 0(x26) — misaligned load word

    # Misaligned HALFWORD store/load (adresă impară)
    for mis in [1, 3]:
        E(addi(26, 20, 64 + mis))
        E(store(0b001, 2, 26, 0))    # SH misaligned
        E(load(0b001, 25, 26, 0))    # LH misaligned
        E(load(0b101, 25, 26, 0))    # LHU misaligned

    # Misaligned cu offset în instrucțiune
    for off in [1, 2, 3]:
        E(store(0b010, 3, 20, off))  # SW x3, off(x20) — misaligned via imm
        E(load(0b010, 25, 20, off))  # LW x25, off(x20)

    # ── 3. Compressed branches și jumps ──────────────────────────────────
    # C.LI — load immediate în regiștri compressed
    for rd in range(8, 12):
        E(c_li(rd, rd - 8))      # x8=0, x9=1, x10=2, x11=3

    # C.BEQZ: branch if x8==0 (x8=0 din setup → TAKEN)
    # [A+0]: C.BEQZ x8', +4 → taken, sare la A+4
    # [A+2]: NOP 16-bit (C.NOP)
    # [A+4]: landing
    E(c_beqz(0, 4))    # rs1'=0 → x8, offset=+4 (taken dacă x8==0)
    E(c_nop())         # sărit când taken
    E(nop())           # aterizare

    # C.BEQZ not-taken: x9=1 ≠ 0
    E(c_beqz(1, 4))    # rs1'=1 → x9=1, not taken
    E(c_nop())         # executat (not taken)
    E(nop())

    # C.BNEZ: branch if x9!=0 (x9=1 → TAKEN)
    E(c_bnez(1, 4))    # rs1'=1 → x9=1, taken
    E(c_nop())         # sărit
    E(nop())           # aterizare

    # C.BNEZ not-taken: x8==0
    E(c_bnez(0, 4))    # rs1'=0 → x8=0, not taken
    E(c_nop())         # executat
    E(nop())

    # C.J — jump necondiționat înainte
    # [B+0]: C.J +6 → sare la B+6
    # [B+2]: C.NOP (sărit)
    # [B+4]: C.NOP (sărit)
    # [B+6]: NOP (aterizare)
    E(c_j(6))
    E(c_nop())    # sărit
    E(c_nop())    # sărit
    E(nop())      # aterizare

    # C.JAL — jump and link (RV32 only, linkează x1)
    E(c_jal(6))
    E(c_nop())    # continuare după return (x1=PC+2)
    E(c_nop())
    E(nop())      # aterizare C.JAL, x1=adresa de return

    # C.LWSP — load word from stack pointer (x2)
    for off in [0, 4, 8, 12]:
        E(c_lwsp(5, off))    # LW x5, off(sp)

    # ── 4. JALR cu funct3 invalid (illegal_insn path în decoder) ─────────
    # funct3 != 000 pentru JALR → illegal instruction → trap
    for f3 in [1, 2, 3, 4, 5, 6, 7]:
        ill_jalr = ((0 & 0xFFF) << 20) | (1 << 15) | (f3 << 12) | (0 << 7) | 0b1100111
        E(ill_jalr)
        E(nop())  # returned to here after trap

    # ── 5. JALR valid cu diverse offset-uri ──────────────────────────────
    # JAL x1, +16 → subroutine
    # ADDI x25, x0, 77 (body)
    # JALR x0, 0(x1) → return
    E(jal(1, 16))
    E(nop()); E(nop()); E(nop())
    E(addi(25, 0, 77))
    E(jalr(0, 1, 0))

    # JALR cu offset negativ
    E(jal(2, 16))
    E(nop()); E(nop()); E(nop())
    E(addi(25, 0, 55))
    E(jalr(3, 2, 0))     # rd=x3 salvează link addr

    # ── 6. Instrucțiuni compressed ilegale (illegal_instr_o paths) ───────
    # Rezervate / undefined în spec = declanșează illegal_instr_o
    # C.ADDI16SP cu imm=0 → ilegal
    E(pack_c(0b011_0_00010_00000_01))   # C.LUI/ADDI16SP cu imm=0 = ilegal
    E(nop())

    # ── 7. Store/Load cu byte enable pattern-uri diverse ──────────────────
    # Exercitează toate combinațiile de data_be în LSU
    E(lui(27, 0xABCDE))
    E(addi(27, 27, 0x678 & 0xFFF))
    for off in range(0, 64, 4):
        E(store(0b010, 27, 20, 128 + off))
        E(load(0b010, 25, 20, 128 + off))
    for off in range(0, 16, 2):
        E(store(0b001, 27, 20, 256 + off))
        E(load(0b001, 25, 20, 256 + off))
        E(load(0b101, 25, 20, 256 + off))

    # ── 8. Padding ────────────────────────────────────────────────────────
    for _ in range(16):
        E(nop())

    return p


def run_targeted():
    prog = build_targeted_program()
    print(f"Targeted program: {len(prog)} instructions ({len(prog)*4} bytes)")

    payload = {"n": len(prog), "agent": "l8_targeted",
               "machine_code": [int(w) for w in prog]}
    with open(PROGRAM_JSON, "w") as f:
        json.dump(payload, f)

    import sysconfig as _sc
    _pylib = _sc.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu"
        + ((":" + _pylib) if _pylib else "")
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = PROGRAM_JSON

    print(f"Running Vtop @ {VTOP} ...")
    r = subprocess.run([str(VTOP)], cwd=str(ML4DV), env=env,
                       capture_output=True, text=True, timeout=600)
    print("[OK]" if r.returncode == 0 else f"[WARN] rc={r.returncode}")

    s = cov_parser.parse(str(COVDAT))
    print("\nCoverage (single run):")
    for k in ("toggle", "branch", "line"):
        c, t = s.by_kind[k]
        print(f"  {k:7s}: {c:>5}/{t:>5}  ({100*c/t:.2f}%)")

    print("\nTop modulele acoperite (line, single run):")
    rows = sorted([(c/max(t,1), pg.replace("v_line/",""), c, t)
                   for pg,(c,t) in s.by_page.items()
                   if pg.startswith("v_line/") and t > 0], reverse=True)
    for pct, mod, c, t in rows[:12]:
        print(f"  {100*pct:5.1f}%  {c:>4}/{t:<4}  {mod}")


if __name__ == "__main__":
    run_targeted()
