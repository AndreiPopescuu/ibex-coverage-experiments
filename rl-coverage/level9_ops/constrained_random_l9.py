"""constrained_random_l9.py — Constrained Random baseline pentru L9 toggle coverage.

Față de uniform random:
  1. Diversitate de ops — evită să repete același opcode consecutiv
  2. Probabilități non-uniforme pe ops — branches/loads/R-type boostate
  3. Operanzi extremi mai des — x0, x16(-1), x17(INT_MAX), x18(INT_MIN)
  4. RAW hazard pattern — la fiecare 4 pași: producer → consumer
  5. same_src — din când în când rs1==rs2 (toggle SAME_SRC paths)
  6. Boost ops noi L9 — MRET, WFI, FENCE_I, C_BEQZ, C_BNEZ

Usage:
    python constrained_random_l9.py
    python constrained_random_l9.py --episodes 1200 --steps 256
"""

import argparse, random, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l9_v2 import IbexL9V2Env
from codec_l9 import Op, N_OPS, IMM_BUCKETS


# ── Probabilități pe ops ─────────────────────────────────────────────────────

_OP_PROBS = np.ones(N_OPS, dtype=np.float64)

# R-type ALU → activează paths diverse în ibex_alu + RAW hazards
for op in [Op.ADD, Op.SUB, Op.SLL, Op.SLT, Op.SLTU,
           Op.XOR, Op.SRL, Op.SRA, Op.OR, Op.AND]:
    _OP_PROBS[int(op)] *= 2.0

# M-type → multi-cycle, ibex_multdiv_fast paths
for op in [Op.MUL, Op.MULH, Op.MULHSU, Op.MULHU,
           Op.DIV, Op.DIVU, Op.REM, Op.REMU]:
    _OP_PROBS[int(op)] *= 1.8

# Branches → taken/not-taken, forward/backward
for op in [Op.BEQ, Op.BNE, Op.BLT, Op.BGE, Op.BLTU, Op.BGEU]:
    _OP_PROBS[int(op)] *= 2.5

# Loads/stores → memory paths, misaligned corner cases
for op in [Op.LW, Op.LH, Op.LHU, Op.LB, Op.LBU, Op.SW, Op.SH, Op.SB]:
    _OP_PROBS[int(op)] *= 2.0

# CSR → ibex_cs_registers paths
for op in [Op.CSRRW, Op.CSRRS, Op.CSRRC,
           Op.CSRRWI, Op.CSRRSI, Op.CSRRCI]:
    _OP_PROBS[int(op)] *= 1.5

# Ops noi L9 — targetează paths complet neacoperite
for op in [Op.MRET, Op.WFI, Op.FENCE_I]:
    _OP_PROBS[int(op)] *= 3.0   # boost mare: paths unice în ibex_controller

# Compressed branches noi
for op in [Op.C_BEQZ, Op.C_BNEZ, Op.C_J, Op.C_JAL]:
    _OP_PROBS[int(op)] *= 2.0

_OP_PROBS /= _OP_PROBS.sum()

# ── Probabilități pe regiștri ─────────────────────────────────────────────────

_REG_PROBS = np.ones(32, dtype=np.float64)
_REG_PROBS[0]  *= 3.0   # x0 → RD_ZERO + RS_ZERO
_REG_PROBS[16] *= 2.0   # x16 = -1
_REG_PROBS[17] *= 2.0   # x17 = INT_MAX
_REG_PROBS[18] *= 2.0   # x18 = INT_MIN
_REG_PROBS[19] *= 1.5   # x19 = 0x55555555 (alternating bits)
_REG_PROBS[21] *= 1.5   # x21 = 0xAAAAAAAA
_REG_PROBS /= _REG_PROBS.sum()

# ── Probabilități pe imm_bucket ───────────────────────────────────────────────
# Bucket 0 = zero, bucket 4 = extrem → corner cases
_IMM_PROBS = np.array([3.0, 1.0, 1.0, 1.0, 3.0])
_IMM_PROBS /= _IMM_PROBS.sum()

# Ops R-type pentru RAW hazard pattern (producer)
_R_TYPE = [int(op) for op in [
    Op.ADD, Op.SUB, Op.XOR, Op.OR, Op.AND,
    Op.SLL, Op.SRL, Op.SRA, Op.SLT, Op.SLTU,
    Op.MUL, Op.DIV, Op.REM, Op.REMU,
]]

# Ops consumer pentru RAW hazard
_CONSUMER = [int(op) for op in [
    Op.ADD, Op.SUB, Op.XOR, Op.OR, Op.AND,
    Op.MUL, Op.ADDI, Op.LW, Op.SW, Op.BEQ, Op.BNE,
]]


class ConstrainedRandomAgent:

    def __init__(self):
        self._step    = 0
        self._last_op = -1
        self._last_rd = -1

    def reset(self):
        self._step    = 0
        self._last_op = -1
        self._last_rd = -1

    def sample(self) -> tuple:
        self._step += 1

        # RAW hazard pattern: la fiecare 4 pași
        if self._step % 4 == 1:
            op_i = random.choice(_R_TYPE)
            rd   = random.randint(1, 31)
            rs1  = int(np.random.choice(32, p=_REG_PROBS))
            rs2  = int(np.random.choice(32, p=_REG_PROBS))
            imm  = int(np.random.choice(IMM_BUCKETS, p=_IMM_PROBS))
            self._last_rd = rd
            self._last_op = op_i
            return (op_i, rd, rs1, rs2, imm)

        if self._step % 4 == 2 and self._last_rd > 0:
            op_i = random.choice(_CONSUMER)
            if random.random() < 0.5:
                rs1, rs2 = self._last_rd, int(np.random.choice(32, p=_REG_PROBS))
            else:
                rs1, rs2 = int(np.random.choice(32, p=_REG_PROBS)), self._last_rd
            rd  = int(np.random.choice(32, p=_REG_PROBS))
            imm = int(np.random.choice(IMM_BUCKETS, p=_IMM_PROBS))
            self._last_rd = -1
            return (int(op_i), rd, rs1, rs2, imm)

        # Op diversity — penalizează repetiția
        op_probs = _OP_PROBS.copy()
        if self._last_op >= 0:
            op_probs[self._last_op] *= 0.1
            op_probs /= op_probs.sum()

        op_i = int(np.random.choice(N_OPS, p=op_probs))
        self._last_op = op_i

        rd  = int(np.random.choice(32, p=_REG_PROBS))
        rs1 = int(np.random.choice(32, p=_REG_PROBS))
        rs2 = int(np.random.choice(32, p=_REG_PROBS))
        imm = int(np.random.choice(IMM_BUCKETS, p=_IMM_PROBS))

        # same_src — 10% șanse rs1==rs2
        if random.random() < 0.1:
            rs2 = rs1

        return (op_i, rd, rs1, rs2, imm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="l9_constrained_random_curve.npz")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 64)
    print(f"L9 Constrained Random — {args.episodes} ep × {args.steps} steps, 83 ops")
    print("=" * 64)
    print(f"  Strategii: op_diversity + RAW_hazard + extreme_operands + same_src")
    print(f"  Boost ops noi: MRET×3, WFI×3, FENCE_I×3, C_BEQZ×2, C_BNEZ×2")
    print(f"\n{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'wall':>5}")
    print("-" * 42)

    env   = IbexL9V2Env(episode_steps=args.steps, seed=args.seed)
    agent = ConstrainedRandomAgent()

    ep_pcts  = []
    cum_pcts = []

    for ep in range(1, args.episodes + 1):
        env.reset()
        agent.reset()
        t0 = time.time()

        for _ in range(args.steps):
            action = agent.sample()
            _, _, term, trunc, info = env.step(action)
            if term or trunc:
                break

        dt = time.time() - t0
        ep_pct  = info.get("ep_pct",  0.0)
        cum_pct = info.get("cum_pct", 0.0)
        new     = info.get("new_hits_vs_cum", 0)
        ep_pcts.append(ep_pct)
        cum_pcts.append(cum_pct)

        print(f"  ep {ep:>4} | ep {ep_pct:>5.2f}% | cum {cum_pct:>5.2f}% | "
              f"new {new:>4} | {dt:.1f}s", flush=True)

    print(f"\nRezultate finale:")
    print(f"  Constrained Random best:  {max(cum_pcts):.2f}%")
    print(f"  Constrained Random final: {cum_pcts[-1]:.2f}%")
    print(f"  L8 random baseline:       66.07%  (referință)")

    eps = np.arange(1, len(ep_pcts) + 1)
    np.savez(args.out,
             ep=eps,
             ep_pct=np.array(ep_pcts),
             cum_pct=np.array(cum_pcts))
    print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
