"""env_l8_guided.py — L8 env cu observation space îmbogățit și reward shaping.

Față de env_l8.py:
  - Observation space: 10 features (step_frac, cum_cov, 8 per-modul fractions)
  - Reward shaping: hits în module slab acoperite valorează mai mult
  - Acceptă initial_hits (set pre-acumulat din CDG)

Module urmărite (cele cu coverage scăzut):
  ibex_counter, ibex_cs_registers, ibex_if_stage, ibex_controller,
  ibex_csr, ibex_core, ibex_alu, ibex_multdiv_fast
"""

import os, sys, re, json, subprocess
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS  = Path(__file__).resolve().parent
L5    = (THIS.parent / "level5_real_rtl").resolve()
ML4DV = (THIS.parent.parent / "cpu").resolve()
VTOP  = ML4DV / "sim_build" / "Vtop"
COVDAT = ML4DV / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l8_guided.json"

sys.path.insert(0, str(L5))
import cov_parser

from codec_l8 import N_OPS, IMM_BUCKETS, emit_program

# ── Module de urmărit + reward weights ───────────────────────────────────

MODULES = [
    "ibex_counter",        # 22.7% → weight 4x
    "ibex_cs_registers",   # 29.0% → weight 3x
    "ibex_if_stage",       # 40.6% → weight 2x
    "ibex_controller",     # 42.5% → weight 2x
    "ibex_csr",            # 46.2% → weight 2x
    "ibex_core",           # 59.7% → weight 1.5x
    "ibex_alu",            # 60.0% → weight 1.5x
    "ibex_multdiv_fast",   # ~60%  → weight 1.5x
]

MODULE_WEIGHTS = {
    "ibex_counter":      4.0,
    "ibex_cs_registers": 3.0,
    "ibex_if_stage":     2.0,
    "ibex_controller":   2.0,
    "ibex_csr":          2.0,
    "ibex_core":         1.5,
    "ibex_alu":          1.5,
    "ibex_multdiv_fast": 1.5,
}
DEFAULT_WEIGHT = 1.0

N_OBS = 2 + len(MODULES)   # step_frac + cum_cov + 8 module fractions


def module_of(key: str) -> str | None:
    """Extrage numele modulului dintr-un coverage point key."""
    m = re.search(r"page\x02v_toggle/([^\x01]+)\x01", key)
    if not m:
        return None
    page = m.group(1)
    return page.split("__")[0]


def run_program(actions):
    machine = emit_program(actions)
    with open(PROGRAM_JSON, "w") as f:
        json.dump({"n": len(machine), "agent": "l8",
                   "machine_code": [int(m) for m in machine]}, f)
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


class IbexL8GuidedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 1024, seed: int | None = None,
                 initial_hits: set | None = None):
        super().__init__()
        self.episode_steps = episode_steps
        self.action_space = spaces.MultiDiscrete(
            [N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS,), dtype=np.float32)

        self._actions    = []
        self._step_idx   = 0
        self._n_episodes = 0

        # Hits cumulativi pre-acumulați (din CDG)
        self._cum_hits = set(initial_hits) if initial_hits else set()

        # Per-modul: {mod: [covered_cum, total]}
        self._mod_covered = {m: 0 for m in MODULES}
        self._mod_total   = {m: 1 for m in MODULES}  # evitam div/0
        self._total_tog   = 1
        self._key_to_mod  = {}  # cache key→modul (populat la primul run)

        # Pre-populăm per-modul dacă avem hits inițiale
        if self._cum_hits:
            for key in self._cum_hits:
                mod = module_of(key)
                self._key_to_mod[key] = mod
                if mod and mod in self._mod_covered:
                    self._mod_covered[mod] += 1

    def _update_mod_counts_from_hits(self, hits: set):
        """Actualizează contoarele per-modul din setul de hits."""
        for key in hits:
            mod = self._key_to_mod.get(key)
            if mod and mod in self._mod_covered:
                self._mod_covered[mod] += 1

    def _obs(self) -> np.ndarray:
        step_frac = self._step_idx / self.episode_steps
        cum_cov   = len(self._cum_hits) / max(self._total_tog, 1)
        mod_fracs = [
            self._mod_covered[m] / max(self._mod_total[m], 1)
            for m in MODULES
        ]
        return np.array([step_frac, cum_cov] + mod_fracs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._actions.clear()
        self._step_idx = 0
        return self._obs(), {}

    def step(self, action):
        self._actions.append(tuple(int(x) for x in action))
        self._step_idx += 1
        truncated = self._step_idx >= self.episode_steps
        reward, info = 0.0, {}

        if truncated:
            summary = run_program(self._actions)
            if summary is None:
                info["vtop_failed"] = True
            else:
                toggle_covered, toggle_total = summary.by_kind["toggle"]
                self._total_tog = max(toggle_total, 1)

                prefix = "\x01page\x02v_toggle/"
                ep_hits = {k for k, v in summary.points.items()
                           if v > 0 and prefix in ("\x01" + k)}
                new_hits = ep_hits - self._cum_hits
                self._cum_hits |= ep_hits

                # Actualizează cache key→modul și contoare incremental
                for key in ep_hits:
                    if key not in self._key_to_mod:
                        self._key_to_mod[key] = module_of(key)
                    mod = self._key_to_mod[key]
                    if mod and mod in self._mod_covered and key in new_hits:
                        self._mod_covered[mod] += 1

                # Actualizează totaluri per-modul (o singură dată la primul run)
                if not hasattr(self, "_mod_total_set"):
                    self._mod_total_set = True
                    from collections import Counter
                    mod_counts = Counter()
                    for key in summary.points:
                        if key not in self._key_to_mod:
                            self._key_to_mod[key] = module_of(key)
                        mod = self._key_to_mod[key]
                        if mod:
                            mod_counts[mod] += 1
                    for mod in MODULES:
                        if mod in mod_counts:
                            self._mod_total[mod] = mod_counts[mod]

                # Reward shaping: hits noi ponderate per modul
                shaped_reward = 0.0
                for key in new_hits:
                    mod = module_of(key)
                    w = MODULE_WEIGHTS.get(mod, DEFAULT_WEIGHT)
                    shaped_reward += w

                reward = shaped_reward

                cum_pct = 100.0 * len(self._cum_hits) / self._total_tog
                info.update({
                    "ep_pct":         100.0 * toggle_covered / self._total_tog,
                    "ep_total":       toggle_total,
                    "cum_covered":    len(self._cum_hits),
                    "cum_pct":        cum_pct,
                    "new_hits_vs_cum": len(new_hits),
                    "shaped_reward":  shaped_reward,
                    "branch_pct":     100.0 * summary.by_kind["branch"][0] /
                                      max(summary.by_kind["branch"][1], 1),
                    "mod_coverage":   {m: self._mod_covered[m] /
                                       max(self._mod_total[m], 1)
                                       for m in MODULES},
                })
            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info
