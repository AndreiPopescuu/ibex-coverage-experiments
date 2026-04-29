"""eval_on_verilator.py — Evaluare model shadow pe Verilator real.

Reconstruiește obs 14-dim (compatibil cu IbexL9ShadowEnv) folosind
Verilator ca simulator în loc de shadow.

Usage:
    python eval_on_verilator.py --episodes 2000
    python eval_on_verilator.py --episodes 2000 --model l9_shadow_final
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from env_l9 import IbexL9Env, run_program
from env_l9_shadow import (
    _TOTAL_BINS, _SEEN_T, _OPAIR_T, _RAW_T, _SEQ_T, _CORNER_T,
    _RDZERO_T, _RS1ZERO_T, _IMM_T, _SAMESRC_T,
    _SEQ_NAME_TO_PAIR, _RAW_NAME_TO_TRIPLE,
    _N_PRODUCERS, _RAW_CONSUMER_TOTAL,
    N_OPS, N_INSTRS, INSTR_NAMES, HAS_RS1, HAS_RS2, HAS_RD,
)

try:
    from stable_baselines3 import PPO
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


class VerilatorObs14:
    """Reconstruiește obs 14-dim pentru un model antrenat pe shadow."""

    def __init__(self, episode_steps: int, max_episodes: int):
        self.episode_steps = episode_steps
        self.max_episodes  = max_episodes

        self._cum_hits      : set = set()
        self._cum_seq_pairs : set = set()
        self._cum_raw_pairs : set = set()
        self._cat = {"seen": 0, "opair": 0, "raw": 0, "seq": 0, "corner": 0,
                     "rdzero": 0, "rs1zero": 0, "imm": 0, "samesrc": 0}

        self._step_idx   = 0
        self._n_episodes = 0
        self._last_op    = 0
        self._last_rd    = 0
        self._recent_rds : list = []
        self._recent_ops : list = []

    def reset(self):
        self._step_idx = 0
        self._last_op  = 0
        self._last_rd  = 0
        self._recent_rds.clear()
        self._recent_ops.clear()
        # Precompute obs[12] și [13] pentru toate op_i (se schimbă doar la end-of-episode)
        self._seq_frac_cache = np.array(
            [self._seq_covered_frac(i) for i in range(N_OPS)], dtype=np.float32)
        self._raw_frac_cache = np.array(
            [self._raw_uncov_consumer_frac(i) for i in range(N_OPS)], dtype=np.float32)

    def update_step(self, op_i: int, rd: int):
        self._step_idx += 1
        if rd != 0:
            self._recent_rds.append(rd)
            self._recent_ops.append(op_i)
            if len(self._recent_rds) > 3:
                self._recent_rds.pop(0)
                self._recent_ops.pop(0)
        self._last_op = op_i
        self._last_rd = rd

    def update_episode(self, new_hits: set):
        for h in new_hits:
            if   h.startswith("seen_"):    self._cat["seen"]    += 1
            elif h.startswith("opair_"):   self._cat["opair"]   += 1
            elif h.startswith("raw"):      self._cat["raw"]     += 1
            elif h.startswith("seq_"):     self._cat["seq"]     += 1
            elif h.startswith("corner_"):  self._cat["corner"]  += 1
            elif h.startswith("rdzero_"):  self._cat["rdzero"]  += 1
            elif h.startswith("rs1zero_"): self._cat["rs1zero"] += 1
            elif h.startswith("imm_"):     self._cat["imm"]     += 1
            elif h.startswith("samesrc_"): self._cat["samesrc"] += 1
            pair = _SEQ_NAME_TO_PAIR.get(h)
            if pair: self._cum_seq_pairs.add(pair)
            triple = _RAW_NAME_TO_TRIPLE.get(h)
            if triple: self._cum_raw_pairs.add(triple)
        self._cum_hits |= new_hits
        self._n_episodes += 1

    def _seq_covered_frac(self, op_i: int) -> float:
        if op_i >= N_INSTRS:
            return 0.0
        covered = sum(1 for j in range(N_INSTRS) if (op_i, j) in self._cum_seq_pairs)
        return covered / N_INSTRS

    def _raw_uncov_consumer_frac(self, op_i: int) -> float:
        if op_i >= N_INSTRS:
            return 0.0
        name = INSTR_NAMES[op_i]
        if name not in HAS_RS1 and name not in HAS_RS2:
            return 0.0
        covered = sum(
            1 for d in range(3) for i in range(N_INSTRS)
            if INSTR_NAMES[i] in HAS_RD and (d, i, op_i) in self._cum_raw_pairs
        )
        return 1.0 - min(1.0, covered / _RAW_CONSUMER_TOTAL)

    def obs(self) -> np.ndarray:
        op_i = self._last_op
        return np.array([
            self._step_idx / max(self.episode_steps, 1),
            min(1.0, len(self._cum_hits) / _TOTAL_BINS),
            min(1.0, self._cat["seen"]   / max(_SEEN_T,    1)),
            min(1.0, self._cat["opair"]  / max(_OPAIR_T,   1)),
            min(1.0, self._cat["raw"]    / max(_RAW_T,     1)),
            min(1.0, self._cat["seq"]    / max(_SEQ_T,     1)),
            min(1.0, self._cat["corner"] / max(_CORNER_T,  1)),
            0.0,
            self._last_op / max(N_OPS - 1, 1),
            min(1.0, self._n_episodes / max(self.max_episodes, 1)),
            self._last_rd / 31.0,
            float(len(self._recent_rds)) / 3.0,
            self._seq_frac_cache[op_i] if op_i < N_OPS else 0.0,
            self._raw_frac_cache[op_i] if op_i < N_OPS else 0.0,
            min(1.0, self._cat["rdzero"]  / max(_RDZERO_T,  1)),
            min(1.0, self._cat["rs1zero"] / max(_RS1ZERO_T, 1)),
            min(1.0, self._cat["imm"]     / max(_IMM_T,     1)),
            min(1.0, self._cat["samesrc"] / max(_SAMESRC_T, 1)),
        ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--steps",    type=int, default=480)
    ap.add_argument("--model",    default="l9_shadow_final")
    ap.add_argument("--driver",   default="test_run_for_l9")
    args = ap.parse_args()

    model = PPO.load(str(THIS / args.model))
    print(f"Model: {args.model}")
    print(f"Evaluare: {args.episodes} ep × {args.steps} pași pe Verilator")
    print(f"{'ep':>6}  {'ep%':>7}  {'cum%':>8}  {'new':>6}  {'s':>5}")
    print("-" * 40)

    tracker = VerilatorObs14(episode_steps=args.steps, max_episodes=args.episodes)
    ep_pcts = []; cum_pcts = []; new_list = []
    t_total = time.time()

    for ep in range(1, args.episodes + 1):
        tracker.reset()
        actions = []
        t0 = time.time()

        for step in range(args.steps):
            obs = tracker.obs()
            action, _ = model.predict(obs, deterministic=False)
            op_i, rd, rs1, rs2, imm_b = (int(x) for x in action)
            actions.append((op_i, rd, rs1, rs2, imm_b))
            tracker.update_step(op_i, rd)

        result = run_program(actions, driver=args.driver)
        if result is None:
            print(f"{ep:>6}  [VTOP FAIL]")
            continue

        ep_hits  = set(result["hits"])
        new_hits = ep_hits - tracker._cum_hits
        tracker.update_episode(new_hits)

        ep_pct  = result["pct"]
        cum_pct = 100.0 * len(tracker._cum_hits) / _TOTAL_BINS
        new_n   = len(new_hits)
        elapsed = int(time.time() - t0)

        ep_pcts.append(ep_pct)
        cum_pcts.append(cum_pct)
        new_list.append(new_n)

        if ep % 100 == 0:
            avg_new = sum(new_list[-100:]) / 100
            print(f"  ── ep {ep}: cum={cum_pct:.2f}%  avg_new/ep={avg_new:.2f} ──")

        print(f"{ep:>6}  {ep_pct:>6.1f}%  {cum_pct:>7.2f}%  {new_n:>6}  {elapsed:>4}s")

    total_time = time.time() - t_total
    print("=" * 40)
    print(f"Final cum coverage (Verilator): {cum_pcts[-1]:.2f}%")
    print(f"Total bins: {_TOTAL_BINS}")
    print(f"Wall time: {total_time:.0f}s")

    save = THIS / f"l9_ppo_verilator_{args.episodes}ep.npz"
    np.savez(str(save),
             ep_pct  = np.array(ep_pcts),
             cum_pct = np.array(cum_pcts),
             new_hits= np.array(new_list))
    print(f"[saved] {save}")


if __name__ == "__main__":
    main()
