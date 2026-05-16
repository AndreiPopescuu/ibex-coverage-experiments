"""train_l9_rtl.py — PPO antrenat direct pe Verilator (RTL real), obs 18-dim.

Problema cu train_l9.py original: obs space 6-dim cu 3 campuri mereu zero
→ PPO invata orb. Acest script foloseste acelasi obs space 18-dim ca shadow.

Comparatie tinta:
  - Constrained random (2000 ep Verilator): 62.88%
  - PPO vechi (2000 ep Verilator): 59.3%  ← obs space defect
  - Acest script (3000 ep Verilator): ???  ← obs space corect

Usage:
    python train_l9_rtl.py
    python train_l9_rtl.py --episodes 2000
    python train_l9_rtl.py --sim-build sim_build_opentitan
"""

import argparse, sys, time
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from env_l9 import run_program
from env_l9_shadow import (
    _TOTAL_BINS, _SEEN_T, _OPAIR_T, _RAW_T, _SEQ_T, _CORNER_T,
    _RDZERO_T, _RS1ZERO_T, _IMM_T, _SAMESRC_T,
    _SEQ_NAME_TO_PAIR, _RAW_NAME_TO_TRIPLE,
    _N_PRODUCERS, _RAW_CONSUMER_TOTAL,
)
from coverage_model import N_INSTRS, INSTR_NAMES, HAS_RD, HAS_RS1, HAS_RS2
from codec_l8 import N_OPS, IMM_BUCKETS

try:
    from stable_baselines3 import PPO
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


EPISODE_STEPS  = 480
TOTAL_EPISODES = 3000


class IbexL9RTLEnv(gym.Env):
    """Gym env cu obs 18-dim antrenat direct pe Verilator."""

    metadata = {"render_modes": []}

    def __init__(self, episode_steps=480, max_episodes=3000, driver="test_run_for_l9"):
        super().__init__()
        self.episode_steps = episode_steps
        self.max_episodes  = max_episodes
        self.driver        = driver

        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )

        # Stare cumulata (persist intre episoade)
        self._cum_hits      : set = set()
        self._cum_seq_pairs : set = set()
        self._cum_raw_pairs : set = set()
        self._cat = {"seen": 0, "opair": 0, "raw": 0, "seq": 0, "corner": 0,
                     "rdzero": 0, "rs1zero": 0, "imm": 0, "samesrc": 0}
        self._n_episodes = 0

        # Stare per-episod
        self._actions    : list = []
        self._step_idx   : int  = 0
        self._last_op    : int  = 0
        self._last_rd    : int  = 0
        self._recent_rds : list = []
        self._recent_ops : list = []

        # Cache obs[12] si obs[13] (actualizate la end-of-episode)
        self._seq_frac_cache = np.zeros(N_OPS, dtype=np.float32)
        self._raw_frac_cache = np.zeros(N_OPS, dtype=np.float32)

    def _seq_covered_frac(self, op_i: int) -> float:
        if op_i >= N_INSTRS: return 0.0
        covered = sum(1 for j in range(N_INSTRS) if (op_i, j) in self._cum_seq_pairs)
        return covered / N_INSTRS

    def _raw_uncov_consumer_frac(self, op_i: int) -> float:
        if op_i >= N_INSTRS: return 0.0
        name = INSTR_NAMES[op_i]
        if name not in HAS_RS1 and name not in HAS_RS2: return 0.0
        covered = sum(
            1 for d in range(3) for i in range(N_INSTRS)
            if INSTR_NAMES[i] in HAS_RD and (d, i, op_i) in self._cum_raw_pairs
        )
        return 1.0 - min(1.0, covered / _RAW_CONSUMER_TOTAL)

    def _obs(self) -> np.ndarray:
        op_i = self._last_op
        return np.array([
            self._step_idx / max(self.episode_steps, 1),               # [0]
            min(1.0, len(self._cum_hits) / _TOTAL_BINS),               # [1]
            min(1.0, self._cat["seen"]   / max(_SEEN_T,    1)),        # [2]
            min(1.0, self._cat["opair"]  / max(_OPAIR_T,   1)),        # [3]
            min(1.0, self._cat["raw"]    / max(_RAW_T,     1)),        # [4]
            min(1.0, self._cat["seq"]    / max(_SEQ_T,     1)),        # [5]
            min(1.0, self._cat["corner"] / max(_CORNER_T,  1)),        # [6]
            0.0,                                                         # [7] ep_new (necunoscut intre pasi)
            self._last_op / max(N_OPS - 1, 1),                         # [8]
            min(1.0, self._n_episodes / max(self.max_episodes, 1)),    # [9]
            self._last_rd / 31.0,                                      # [10]
            float(len(self._recent_rds)) / 3.0,                        # [11]
            self._seq_frac_cache[op_i] if op_i < N_OPS else 0.0,      # [12]
            self._raw_frac_cache[op_i] if op_i < N_OPS else 0.0,      # [13]
            min(1.0, self._cat["rdzero"]  / max(_RDZERO_T,  1)),       # [14]
            min(1.0, self._cat["rs1zero"] / max(_RS1ZERO_T, 1)),       # [15]
            min(1.0, self._cat["imm"]     / max(_IMM_T,     1)),       # [16]
            min(1.0, self._cat["samesrc"] / max(_SAMESRC_T, 1)),       # [17]
        ], dtype=np.float32)

    def _update_cat_cache(self, new_hits: set):
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

    def reset(self, *, seed=None, options=None):
        self._actions.clear()
        self._step_idx = 0
        self._last_op  = 0
        self._last_rd  = 0
        self._recent_rds.clear()
        self._recent_ops.clear()
        return self._obs(), {}

    def step(self, action):
        op_i, rd, rs1, rs2, imm_b = (int(x) for x in action)
        self._actions.append((op_i, rd, rs1, rs2, imm_b))
        self._step_idx += 1

        if rd != 0:
            self._recent_rds.append(rd)
            self._recent_ops.append(op_i)
            if len(self._recent_rds) > 3:
                self._recent_rds.pop(0)
                self._recent_ops.pop(0)
        self._last_op = op_i
        self._last_rd = rd

        truncated = self._step_idx >= self.episode_steps
        reward = 0.0
        info   = {}

        if truncated:
            result = run_program(self._actions, driver=self.driver)
            if result is None:
                reward = -5.0
            else:
                ep_hits  = set(result["hits"])
                new_hits = ep_hits - self._cum_hits
                self._update_cat_cache(new_hits)
                self._cum_hits |= ep_hits

                reward = float(len(new_hits)) * 10.0
                if len(new_hits) == 0:
                    reward = -1.0

                cum_pct = 100.0 * len(self._cum_hits) / _TOTAL_BINS
                info.update({
                    "ep_pct"  : result["pct"],
                    "cum_pct" : cum_pct,
                    "new_hits": len(new_hits),
                })

                # Actualizeaza cache-ul obs[12] si obs[13]
                for i in range(min(N_OPS, N_INSTRS)):
                    self._seq_frac_cache[i] = self._seq_covered_frac(i)
                    self._raw_frac_cache[i] = self._raw_uncov_consumer_frac(i)

            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",   type=int, default=TOTAL_EPISODES)
    ap.add_argument("--steps",      type=int, default=EPISODE_STEPS)
    ap.add_argument("--driver",     default="test_run_for_l9")
    ap.add_argument("--sim-build",  default=None,
                    help="Override IBEX_SIM_BUILD (e.g. sim_build_opentitan)")
    ap.add_argument("--dry-run",    type=int, default=0, metavar="N",
                    help="Ruleaza N episoade de test si estimeaza durata totala")
    args = ap.parse_args()

    if args.sim_build:
        import os
        os.environ["IBEX_SIM_BUILD"] = args.sim_build

    if args.dry_run > 0:
        print(f"=== DRY RUN: {args.dry_run} episoade de test ===")
        import random as _rnd
        env_dry = IbexL9RTLEnv(episode_steps=args.steps, driver=args.driver)
        times = []
        for i in range(1, args.dry_run + 1):
            t0 = time.time()
            actions = [(int(_rnd.randint(0, N_OPS-1)), int(_rnd.randint(0, 31)),
                        int(_rnd.randint(0, 31)), int(_rnd.randint(0, 31)),
                        int(_rnd.randint(0, 4))) for _ in range(args.steps)]
            result = run_program(actions, driver=args.driver)
            elapsed = time.time() - t0
            times.append(elapsed)
            status = f"{result['pct']:.1f}%" if result else "FAIL"
            print(f"  ep {i}: {elapsed:.1f}s  ({status})")
        avg = sum(times) / len(times)
        total_est = avg * args.episodes
        print(f"\nMedie: {avg:.1f}s/ep")
        print(f"Estimat pentru {args.episodes} ep: "
              f"{total_est/3600:.1f} ore  ({total_est/60:.0f} minute)")
        return

    print("=" * 68)
    print("PPO DIRECT PE VERILATOR — obs 18-dim (fix fata de train_l9.py)")
    print(f"  {args.steps} pasi/ep  x  {args.episodes} episoade")
    print(f"  Total bins: {_TOTAL_BINS}")
    print(f"  Estimat: {args.episodes * 7 / 3600:.1f} ore la ~7s/ep")
    print(f"  Target de batut: constrained random 62.88% (2000 ep)")
    print("=" * 68)

    env   = IbexL9RTLEnv(episode_steps=args.steps, max_episodes=args.episodes,
                          driver=args.driver)
    model = PPO(
        "MlpPolicy", env,
        n_steps       = args.steps,
        batch_size    = 96,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        learning_rate = 2e-4,
        ent_coef      = 0.25,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 0,
        device        = "cpu",
        policy_kwargs = dict(net_arch=[512, 512, 256]),
    )

    ep_pcts  = []; cum_pcts  = []; new_hits_log = []
    t_total  = time.time()

    print(f"\n{'ep':>6}  {'ep%':>7}  {'cum%':>8}  {'new':>6}  {'rew':>8}  {'s':>5}")
    print("-" * 50)

    for ep in range(1, args.episodes + 1):
        t0 = time.time()

        model.learn(
            total_timesteps     = args.steps,
            reset_num_timesteps = (ep == 1),
        )

        obs, _ = env.reset()
        done   = False; ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, rew, _, done, info = env.step(action)
            ep_rew += rew

        ep_pct  = info.get("ep_pct",  0.0)
        cum_pct = info.get("cum_pct", 0.0)
        new_n   = info.get("new_hits", 0)
        elapsed = time.time() - t0

        ep_pcts.append(ep_pct)
        cum_pcts.append(cum_pct)
        new_hits_log.append(new_n)

        if ep % 100 == 0:
            avg_new = sum(new_hits_log[-100:]) / 100
            model.save(str(THIS / f"l9_rtl_checkpoint_ep{ep}"))
            print(f"  ── ep {ep}: cum={cum_pct:.2f}%  avg_new={avg_new:.1f}  "
                  f"[checkpoint]  ETA: {(args.episodes-ep)*7/3600:.1f}h ──")

        print(f"{ep:>6}  {ep_pct:>6.2f}%  {cum_pct:>7.2f}%  "
              f"{new_n:>6}  {ep_rew:>8.1f}  {elapsed:>4.1f}s")

    print("=" * 68)
    print(f"Final RTL cum coverage: {cum_pcts[-1]:.2f}%")
    print(f"Wall time: {time.time() - t_total:.0f}s")
    print(f"Constrained random baseline: 62.88%  →  "
          f"{'BATE CR! ✓' if cum_pcts[-1] > 62.88 else 'sub CR'}")

    save_path = THIS / "l9_rtl_curve.npz"
    np.savez(str(save_path),
             ep_pct   = np.array(ep_pcts),
             cum_pct  = np.array(cum_pcts),
             new_hits = np.array(new_hits_log),
    )
    print(f"[saved] {save_path}")

    model.save(str(THIS / "l9_rtl_final"))
    print(f"[saved] {THIS / 'l9_rtl_final'}.zip")


if __name__ == "__main__":
    main()
