"""constrained_random_l9.py — Constrained Random baseline pentru Rich Functional Coverage.

Diferențe față de pur random:
  1. Diversitate de ops — evită să repete același opcode consecutiv
  2. Operanzi extremi mai des — ZERO și EXTREME au probabilitate mai mare
     pentru rs1/rs2 (mai multe corner case bins și operand_pair bins)
  3. RAW_HAZARD pattern — la fiecare 4 instrucțiuni, forțează explicit
     o pereche producer→consumer (instrucțiune care scrie rd, urmată
     imediat de una care citește același registru)
  4. Diversitate de regiștri — evită să folosească mereu x1,x2,x3

Nu e RL — nu învață din reward. Dar e mai inteligent decât pur random.

Usage:
    python constrained_random_l9.py --episodes 30 --steps 512
"""

import argparse, sys, time, random
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from env_l9 import IbexL9Env, run_program
from coverage_model import (INSTR_NAMES, INSTR_INDEX, HAS_RD, HAS_RS1, HAS_RS2,
                             N_INSTRS, N_OP_CATS)
from codec_l8 import N_OPS, IMM_BUCKETS


# ── Distribuții non-uniforme ──────────────────────────────────────────────

# Probabilități pentru fiecare op (70 ops)
# Instrucțiunile care acoperă mai multe bin-uri primesc probabilitate mai mare
BASE_OP_PROBS = np.ones(N_OPS, dtype=np.float64)

# Boost instrucțiuni care contribuie la multe bins:
# R-type și M-type → OPERAND_PAIR + RAW_HAZARD + SEQUENCE
for name in ["ADD","SUB","SLL","SLT","SLTU","XOR","SRL","SRA","OR","AND",
             "MUL","DIV","REM","REMU","MULH","MULHSU","MULHU","DIVU"]:
    if name in INSTR_INDEX and INSTR_INDEX[name] < N_OPS:
        BASE_OP_PROBS[INSTR_INDEX[name]] *= 2.0

# Boost branch-uri → corner cases (taken/ntaken/backward)
for name in ["BEQ","BNE","BLT","BGE","BLTU","BGEU"]:
    if name in INSTR_INDEX and INSTR_INDEX[name] < N_OPS:
        BASE_OP_PROBS[INSTR_INDEX[name]] *= 2.5

# Boost load/store → misaligned corner cases
for name in ["LW","LH","LHU","SW","SH"]:
    if name in INSTR_INDEX and INSTR_INDEX[name] < N_OPS:
        BASE_OP_PROBS[INSTR_INDEX[name]] *= 2.0

# Boost JAL/JALR → jalr corner cases
for name in ["JAL","JALR"]:
    if name in INSTR_INDEX and INSTR_INDEX[name] < N_OPS:
        BASE_OP_PROBS[INSTR_INDEX[name]] *= 1.5

BASE_OP_PROBS /= BASE_OP_PROBS.sum()

# Probabilități pentru regiștri (0-31)
# Boost x0 (pentru RD_ZERO, RS_ZERO bins) și regiștri mari (diversitate)
REG_PROBS = np.ones(32, dtype=np.float64)
REG_PROBS[0] *= 3.0   # x0 → RD_ZERO + RS_ZERO bins
REG_PROBS[16] *= 2.0  # x16 = -1 → EXTREME bins
REG_PROBS[17] *= 2.0  # x17 = INT_MAX
REG_PROBS[18] *= 2.0  # x18 = INT_MIN
REG_PROBS /= REG_PROBS.sum()

# Probabilități pentru imm_bucket
# Bucket 0 = zero, bucket 4 = extrem → mai multe corner cases
IMM_PROBS = np.array([3.0, 1.0, 1.0, 1.0, 3.0])
IMM_PROBS /= IMM_PROBS.sum()


class ConstrainedRandomAgent:
    """Generator de acțiuni constrained random."""

    def __init__(self, episode_steps: int):
        self.episode_steps = episode_steps
        self._step = 0
        self._last_op = -1
        self._last_rd = -1   # ultimul registru scris
        self._raw_counter = 0  # numărător pentru RAW_HAZARD pattern

    def reset(self):
        self._step = 0
        self._last_op = -1
        self._last_rd = -1
        self._raw_counter = 0

    def sample(self) -> tuple:
        """Generează o acțiune (op_i, rd, rs1, rs2, imm_bucket)."""
        self._step += 1

        # ── Strategia 1: RAW_HAZARD pattern (la fiecare 4 pași) ──────────
        # Pasul 4k+1: emit o instrucțiune producătoare (R-type cu rd != x0)
        # Pasul 4k+2: emit o instrucțiune consumatoare (citește rd-ul anterior)
        if self._step % 4 == 1:
            # Producer: alege un op R-type
            r_type_ops = [INSTR_INDEX[n] for n in
                         ["ADD","SUB","XOR","OR","AND","SLL","SRL","SRA","SLT","SLTU",
                          "MUL","DIV","REM","REMU"]
                         if n in INSTR_INDEX and INSTR_INDEX[n] < N_OPS]
            op_i = random.choice(r_type_ops)
            rd   = random.randint(1, 31)   # rd != x0
            rs1  = np.random.choice(32, p=REG_PROBS)
            rs2  = np.random.choice(32, p=REG_PROBS)
            imm  = np.random.choice(IMM_BUCKETS, p=IMM_PROBS)
            self._last_rd  = rd
            self._last_op  = op_i
            self._raw_counter = 1
            return (int(op_i), int(rd), int(rs1), int(rs2), int(imm))

        if self._step % 4 == 2 and self._last_rd > 0:
            # Consumer: alege un op care citește rs1 sau rs2
            consumer_ops = [INSTR_INDEX[n] for n in
                           ["ADD","SUB","XOR","OR","AND","SLL","SRL","SRA",
                            "MUL","DIV","ADDI","LW","SW","BEQ","BNE"]
                           if n in INSTR_INDEX and INSTR_INDEX[n] < N_OPS]
            op_i = random.choice(consumer_ops)
            # Pune rd-ul anterior ca rs1 sau rs2
            if random.random() < 0.5:
                rs1 = self._last_rd; rs2 = np.random.choice(32, p=REG_PROBS)
            else:
                rs1 = np.random.choice(32, p=REG_PROBS); rs2 = self._last_rd
            rd  = np.random.choice(32, p=REG_PROBS)
            imm = np.random.choice(IMM_BUCKETS, p=IMM_PROBS)
            self._last_rd = -1
            return (int(op_i), int(rd), int(rs1), int(rs2), int(imm))

        # ── Strategia 2: Diversitate de ops ──────────────────────────────
        # Evită să repete același op consecutiv
        op_probs = BASE_OP_PROBS.copy()
        if self._last_op >= 0:
            op_probs[self._last_op] *= 0.1   # penalizare pentru repetiție
            op_probs /= op_probs.sum()

        op_i = np.random.choice(N_OPS, p=op_probs)
        self._last_op = op_i

        # ── Strategia 3: Operanzi extremi mai des ─────────────────────────
        rd  = np.random.choice(32, p=REG_PROBS)
        rs1 = np.random.choice(32, p=REG_PROBS)
        rs2 = np.random.choice(32, p=REG_PROBS)
        imm = np.random.choice(IMM_BUCKETS, p=IMM_PROBS)

        # ── Strategia 4: SAME_SRC bins — din când în când rs1==rs2 ────────
        if random.random() < 0.1:  # 10% șanse
            rs2 = rs1

        return (int(op_i), int(rd), int(rs1), int(rs2), int(imm))


def run_constrained(args):
    env = IbexL9Env(episode_steps=args.steps, driver=args.driver)
    agent = ConstrainedRandomAgent(episode_steps=args.steps)

    print(f"Constrained Random: {args.episodes} ep × {args.steps} pași")
    print(f"  Strategii: op_diversity + RAW_HAZARD_pattern + extreme_operands + same_src")
    print(f"{'ep':>5}  {'ep%':>7}  {'cum%':>8}  {'new':>6}  {'time':>6}")
    print("-" * 42)

    ep_pcts = []; cum_pcts = []; new_list = []
    t_total = time.time()

    for ep in range(1, args.episodes + 1):
        t0 = time.time()
        agent.reset()
        actions = [agent.sample() for _ in range(args.steps)]

        result = run_program(actions, driver=args.driver)
        if result is None:
            print(f"{ep:>5}  [FAIL]")
            continue

        ep_hits = set(result["hits"])
        new_hits = ep_hits - env._cum_hits
        env._cum_hits |= ep_hits
        env._total_bins = max(result["total"], 1)

        ep_pct  = result["pct"]
        cum_pct = 100.0 * len(env._cum_hits) / env._total_bins
        new_n   = len(new_hits)

        ep_pcts.append(ep_pct)
        cum_pcts.append(cum_pct)
        new_list.append(new_n)

        elapsed = int(time.time() - t0)
        print(f"{ep:>5}  {ep_pct:>6.1f}%  {cum_pct:>7.1f}%  "
              f"{new_n:>6}  {elapsed:>5}s")

    total_time = time.time() - t_total
    print("=" * 42)
    print(f"Final cum coverage: {cum_pcts[-1]:.2f}%")
    print(f"Total bins: {env._total_bins}")
    print(f"Wall time: {total_time:.0f}s")

    save = THIS / f"l9_constrained_baseline_{args.steps}steps.npz"
    np.savez(str(save),
             ep_pct  = np.array(ep_pcts),
             cum_pct = np.array(cum_pcts),
             new_hits= np.array(new_list))
    print(f"[saved] {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--steps",    type=int, default=512)
    ap.add_argument("--driver",   default="test_run_for_l9")
    args = ap.parse_args()
    run_constrained(args)
