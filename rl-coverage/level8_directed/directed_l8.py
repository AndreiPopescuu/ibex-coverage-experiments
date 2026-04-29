"""Level 8 — Directed program builder v2 (zero p.pop(), ~1400 instrucțiuni)."""

import json, os, sys, subprocess, argparse
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
import cov_parser

ML4DV  = (THIS.parent.parent / "cpu").resolve()
_SIM_BUILD = os.environ.get("IBEX_SIM_BUILD", "sim_build")
VTOP   = ML4DV / _SIM_BUILD / "Vtop"
COVDAT = ML4DV / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l8_directed.json"

# ── Encodere ─────────────────────────────────────────────────────────────
def addi(rd,rs1,imm): return ((imm&0xFFF)<<20)|((rs1&0x1F)<<15)|(0<<12)|((rd&0x1F)<<7)|0b0010011
def lui(rd,i20): return ((i20&0xFFFFF)<<12)|((rd&0x1F)<<7)|0b0110111
def auipc(rd,i20): return ((i20&0xFFFFF)<<12)|((rd&0x1F)<<7)|0b0010111
def r_type(f7,rs2,rs1,f3,rd): return (f7<<25)|((rs2&0x1F)<<20)|((rs1&0x1F)<<15)|(f3<<12)|((rd&0x1F)<<7)|0b0110011
def m_type(rs2,rs1,f3,rd): return r_type(0b0000001,rs2,rs1,f3,rd)
def load(f3,rd,rs1,imm): return ((imm&0xFFF)<<20)|((rs1&0x1F)<<15)|(f3<<12)|((rd&0x1F)<<7)|0b0000011
def store(f3,rs2,rs1,imm):
    return (((imm>>5)&0x7F)<<25)|((rs2&0x1F)<<20)|((rs1&0x1F)<<15)|(f3<<12)|((imm&0x1F)<<7)|0b0100011
def branch(f3,rs1,rs2,off):
    o=off&0x1FFF
    return (((o>>12)&1)<<31)|(((o>>5)&0x3F)<<25)|((rs2&0x1F)<<20)|((rs1&0x1F)<<15)|(f3<<12)|(((o>>1)&0xF)<<8)|(((o>>11)&1)<<7)|0b1100011
def jal(rd,off):
    o=off&0x1FFFFF
    return (((o>>20)&1)<<31)|(((o>>1)&0x3FF)<<21)|(((o>>11)&1)<<20)|(((o>>12)&0xFF)<<12)|((rd&0x1F)<<7)|0b1101111
def jalr(rd,rs1,off): return ((off&0xFFF)<<20)|((rs1&0x1F)<<15)|(0<<12)|((rd&0x1F)<<7)|0b1100111
def csr_r(f3,rd,rs1,csr): return ((csr&0xFFF)<<20)|((rs1&0x1F)<<15)|(f3<<12)|((rd&0x1F)<<7)|0b1110011
def csr_i(f3,rd,uimm,csr): return ((csr&0xFFF)<<20)|((uimm&0x1F)<<15)|(f3<<12)|((rd&0x1F)<<7)|0b1110011
def shift_i(f7,f3,shamt,rs1,rd): return (f7<<25)|((shamt&0x1F)<<20)|((rs1&0x1F)<<15)|(f3<<12)|((rd&0x1F)<<7)|0b0010011
def nop(): return addi(0,0,0)
def ecall(): return 0x00000073
def ebreak(): return 0x00100073
def fence(): return 0x0FF0000F

SAFE_CSRS=[0x340,0x341,0x342,0x343,0xB00,0xB02,0xF14,0xF11,0x320,0xC00,0xC01,0xC02]

def build_directed_program():
    p=[]
    E=p.append

    # 1. Setup regiștri
    for i in range(1,16): E(addi(i,0,i))
    E(addi(16,0,-1&0xFFF))           # x16=-1
    E(lui(17,0x7FFFF)); E(addi(17,17,-1&0xFFF))  # x17=INT_MAX
    E(lui(18,0x80000))                # x18=INT_MIN
    E(lui(20,0x00010))                # x20=0x10000 (data mem base)
    E(lui(21,0x12345)); E(addi(21,21,0x678))  # x21=0x12345678
    for i in range(22,32): E(addi(i,0,i))

    # 2. LUI diverse
    for imm in [0x00001,0x12345,0xABCDE,0xFFFFF,0x80000,0x7FFFF]:
        E(lui(25,imm))

    # 3. AUIPC diverse
    for imm in [0x00001,0x12345,0xFFFFF,0x80000]:
        E(auipc(25,imm))

    # 4. R-type — toate 10 operații
    for f7,f3 in [(0,0),(0b0100000,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0b0100000,5),(0,6),(0,7)]:
        E(r_type(f7,2,1,f3,25))
        E(r_type(f7,2,1,f3,0))        # rd=x0
        E(r_type(f7,2,0,f3,25))       # rs1=x0
        E(r_type(f7,3,3,f3,25))       # rs1==rs2
        E(r_type(f7,17,18,f3,25))     # INT_MAX op INT_MIN
        E(r_type(f7,16,1,f3,25))      # cu -1

    # 5. I-type — ADDI/SLTI/SLTIU/XORI/ORI/ANDI
    for f3,imm in [(0,5),(2,5),(3,5),(4,0xAA),(6,0xAA),(7,0xAA)]:
        op=0b0010011
        E(((imm&0xFFF)<<20)|((1&0x1F)<<15)|(f3<<12)|((25&0x1F)<<7)|op)
        E(((-1&0xFFF)<<20)|((1&0x1F)<<15)|(f3<<12)|((25&0x1F)<<7)|op)
        E(((0&0xFFF)<<20)|((0&0x1F)<<15)|(f3<<12)|((0&0x1F)<<7)|op)
    for imm in [0,1,-1&0xFFF,100,-100&0xFFF,2047,-2048&0xFFF]:
        E(addi(25,1,imm&0xFFF))

    # 6. Shift imediate
    for f7,f3,shamt in [(0,1,1),(0,1,15),(0,1,31),(0,5,1),(0,5,31),(0b0100000,5,1),(0b0100000,5,31)]:
        E(shift_i(f7,f3,shamt,18,25))
        E(shift_i(f7,f3,shamt,17,25))
        E(shift_i(f7,f3,shamt,1,25))

    # 7. MUL/DIV — toate 8 variante × 5 operanzi
    for f3 in range(8):
        E(m_type(2,1,f3,25))
        E(m_type(3,3,f3,25))
        E(m_type(16,1,f3,25))
        E(m_type(0,1,f3,25))    # divide by zero
        E(m_type(17,18,f3,25))
        E(m_type(21,22,f3,25))

    # 8. Stores
    for i,r in enumerate([1,2,3,16,17]): E(store(2,r,20,i*4))          # SW
    for i,r in enumerate([1,2,3,4]):     E(store(1,r,20,32+i*2))       # SH
    for i in range(8):                   E(store(0,i+1,20,64+i))       # SB

    # 9. Loads
    for i in range(5):  E(load(2,25,20,i*4))          # LW
    for i in range(4):  E(load(1,25,20,32+i*2))       # LH
    for i in range(4):  E(load(5,25,20,32+i*2))       # LHU
    for i in range(8):  E(load(0,25,20,64+i))         # LB
    for i in range(8):  E(load(4,25,20,64+i))         # LBU
    for base in [0x100,0x200,0x400]:
        E(addi(26,0,base&0xFFF))
        E(load(2,25,26,0))

    # 10. JAL forward
    E(jal(1,8)); E(nop()); E(nop())   # forward simplu

    # JAL backward (non-looping):
    # [B+0] JAL x2,+16 → B+16
    # [B+4] NOP  ← aterizare backward
    # [B+8] JAL x3,+12 → B+20 (escape)
    # [B+12] NOP
    # [B+16] JAL x4,-12 → B+4 (BACKWARD)
    # [B+20] NOP
    E(jal(2,16)); E(nop()); E(jal(3,12)); E(nop()); E(jal(4,-12)); E(nop())

    # 11. JALR subroutine
    # [C+0] JAL x1,+16 → C+16, x1=C+4
    # [C+4] NOP  ← continuare după return
    # [C+8..C+12] NOP (săritate)
    # [C+16] ADDI x25,x0,42  (subroutine)
    # [C+20] JALR x0,0(x1) → C+4
    E(jal(1,16)); E(nop()); E(nop()); E(nop())
    E(addi(25,0,42)); E(jalr(0,1,0))

    # JALR cu rd!=x0
    E(jal(2,16)); E(nop()); E(nop()); E(nop())
    E(addi(25,0,99)); E(jalr(3,2,0))

    # 12. Branches × taken + not-taken
    def br_pair(f3,r1t,r2t,r1n,r2n):
        E(addi(10,0,r1t&0xFFF)); E(addi(11,0,r2t&0xFFF))
        E(branch(f3,10,11,8)); E(nop())
        E(addi(10,0,r1n&0xFFF)); E(addi(11,0,r2n&0xFFF))
        E(branch(f3,10,11,8)); E(nop())

    br_pair(0,5,5,3,4)          # BEQ
    br_pair(1,3,4,5,5)          # BNE
    br_pair(4,-1&0xFFF,5,10,5)  # BLT
    br_pair(5,10,5,-1&0xFFF,5)  # BGE
    br_pair(6,1,100,100,1)      # BLTU
    br_pair(7,100,1,1,100)      # BGEU
    br_pair(4,-1&0xFFF,1,1,-1&0xFFF)  # BLT extreme
    br_pair(5,0,0,-1&0xFFF,1)   # BGE equal

    # 13. CSR ops
    for csr in SAFE_CSRS:
        E(csr_r(1,25,1,csr))    # CSRRW
        E(csr_r(2,25,1,csr))    # CSRRS
        E(csr_r(3,25,1,csr))    # CSRRC
        E(csr_r(2,25,0,csr))    # CSRRS read-only
        E(csr_i(5,25,7,csr))    # CSRRWI
        E(csr_i(6,25,3,csr))    # CSRRSI
        E(csr_i(7,25,1,csr))    # CSRRCI

    # 14. ECALL + EBREAK
    for _ in range(3):
        E(ecall()); E(nop())
        E(ebreak()); E(nop())

    # 15. FENCE
    for _ in range(3): E(fence()); E(nop())

    # 16. Compressed via codec_l8
    try:
        from codec_l8 import encode as l8e, Op
        for op_i in [int(Op.C_ADDI),int(Op.C_LI),int(Op.C_LUI),int(Op.C_SLLI),
                     int(Op.C_MV),int(Op.C_ADD),int(Op.C_AND),int(Op.C_OR),
                     int(Op.C_XOR),int(Op.C_SUB),int(Op.C_SRLI),int(Op.C_SRAI),
                     int(Op.C_ANDI),int(Op.C_ADDI4SPN),int(Op.C_LW),int(Op.C_SW)]:
            for ib in range(3):
                E(l8e(op_i,10,11,12,ib))
    except Exception as ex:
        print(f"[WARN] compressed: {ex}")

    # 17. Operanzi mari MUL/DIV
    E(lui(26,0x12345)); E(addi(26,26,0x678))
    E(lui(27,0xFEDCB)); E(addi(27,27,0xA98))
    for f3 in range(8):
        E(m_type(27,26,f3,28)); E(m_type(26,27,f3,28))
        E(m_type(0,26,f3,28)); E(m_type(16,26,f3,28))

    # 18. CSR writes intensive
    for csr_addr in [0x341,0x342,0x343,0x320,0xB00,0xB80]:
        E(lui(25,0xABCDE))
        E(csr_r(1,0,25,csr_addr)); E(csr_r(2,25,0,csr_addr))
        E(csr_r(3,25,1,csr_addr)); E(csr_i(5,25,0x1F,csr_addr))

    # 19. Loads/stores adiționale
    for offset in range(0,64,4):
        E(store(2,21,20,128+offset)); E(load(2,25,20,128+offset))
    for offset in range(0,16,2):
        E(store(1,1,20,256+offset)); E(load(1,25,20,256+offset))
        E(load(5,25,20,256+offset))
    for offset in range(16):
        E(store(0,1,20,320+offset)); E(load(0,25,20,320+offset))
        E(load(4,25,20,320+offset))

    # 20. ALU cu cazuri limită
    E(addi(29,0,31))
    E(r_type(0,29,18,1,25))       # SLL INT_MIN<<31
    E(r_type(0,29,18,5,25))       # SRL INT_MIN>>31
    E(r_type(0b0100000,29,18,5,25)) # SRA INT_MIN>>31
    E(r_type(0,16,17,2,25))       # SLT -1,INT_MAX
    E(r_type(0,17,16,2,25))       # SLT INT_MAX,-1
    E(r_type(0,16,17,3,25))       # SLTU
    E(r_type(0,17,16,3,25))       # SLTU swapped

    # Padding
    for _ in range(32): E(nop())

    return p


def run_directed(dry_run=False):
    prog=build_directed_program()
    print(f"Directed program: {len(prog)} instructions ({len(prog)*4} bytes)")
    if dry_run:
        print("[dry-run] skipping simulation"); return
    payload={"n":len(prog),"agent":"l8_v2","machine_code":[int(w) for w in prog]}
    with open(PROGRAM_JSON,"w") as f: json.dump(payload,f)
    import sysconfig as _sc
    _pylib=_sc.get_config_var("LIBDIR") or ""
    env=os.environ.copy()
    env["LD_LIBRARY_PATH"]="/usr/lib/x86_64-linux-gnu"+(":"+_pylib if _pylib else "")+":"+env.get("LD_LIBRARY_PATH","")
    env["MODULE"]="test_run_for_l8"; env["RL_L8_JSON"]=PROGRAM_JSON
    print(f"Running Vtop @ {VTOP} ...")
    r=subprocess.run([str(VTOP)],cwd=str(ML4DV),env=env,capture_output=True,text=True,timeout=600)
    print("[OK] Simulation complete" if r.returncode==0 else f"[WARN] rc={r.returncode}\n{r.stderr[-1000:]}")
    s=cov_parser.parse(str(COVDAT))
    for k in ("toggle","branch","line"):
        c,t=s.by_kind[k]; print(f"  {k:7s}: {c:>5}/{t:>5}  ({100*c/t:.2f}%)")
    print("\nTop-10 uncovered modules (line):")
    rows=sorted([(t-c,pg.replace("v_line/",""),c,t) for pg,(c,t) in s.by_page.items() if pg.startswith("v_line/")],reverse=True)
    for miss,mod,c,t in rows[:10]: print(f"  {miss:>4} miss  {c:>4}/{t:<4} {100*c/t:.1f}%  {mod}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dry-run",action="store_true")
    run_directed(dry_run=ap.parse_args().dry_run)
