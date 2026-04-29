"""env_l9.py — Gym environment pentru Rich Functional Coverage (~15K bins).

Action space: MultiDiscrete([N_OPS=70, 32, 32, 32, IMM_BUCKETS=5])
Obs space:    Box(6,)  — [step_frac, cum_cov_frac, seen_frac, raw_frac, seq_frac, ep_idx_frac]
Reward:       novelty × 10  (bin-uri noi față de cumulat)
"""

import os, sys, json, subprocess
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from coverage_model import CoverageModel, N_INSTRS
from codec_l8 import N_OPS, IMM_BUCKETS, emit_program

ML4DV     = (THIS.parent.parent / "cpu").resolve()
_SIM_BUILD = os.environ.get("IBEX_SIM_BUILD", "sim_build")
VTOP      = ML4DV / _SIM_BUILD / "Vtop"
PROG_JSON = "/tmp/rl_l9_program.json"
COV_JSON  = "/tmp/rl_l9_coverage.json"


def run_program(actions, driver="test_run_for_l9"):
    machine = emit_program(actions)
    with open(PROG_JSON, "w") as f:
        json.dump({"n": len(machine), "agent": "l9",
                   "machine_code": [int(m) for m in machine]}, f)
    import sysconfig as _sc, sys as _sys
    from pathlib import Path as _Path
    _pylib      = _sc.get_config_var("LIBDIR") or ""
    _venv_root  = str(_Path(_sys.executable).parent.parent)
    _venv_bin   = str(_Path(_sys.executable).parent)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu"
        + ((":" + _pylib) if _pylib else "")
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["VIRTUAL_ENV"] = _venv_root
    env["PATH"]        = _venv_bin + ":" + env.get("PATH", "")
    env["MODULE"]    = driver
    env["RL_L9_JSON"] = PROG_JSON
    env["RL_L9_COV"]  = COV_JSON
    proc = subprocess.run(
        [str(VTOP)], cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
    )
    if proc.returncode != 0:
        return None
    try:
        with open(COV_JSON) as f:
            return json.load(f)
    except Exception:
        return None


class IbexL9Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps=256, driver="test_run_for_l9"):
        super().__init__()
        self.episode_steps = episode_steps
        self.driver        = driver

        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self._actions    : list = []
        self._step_idx   : int  = 0
        self._n_episodes : int  = 0
        self._cum_hits   : set  = set()
        self._total_bins : int  = 1

    def _obs(self):
        return np.array([
            self._step_idx / self.episode_steps,
            min(1.0, len(self._cum_hits) / max(self._total_bins, 1)),
            0.0, 0.0, 0.0,
            min(1.0, self._n_episodes / 200),
        ], dtype=np.float32)

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
            result = run_program(self._actions, driver=self.driver)
            if result is None:
                reward = -5.0
                info["vtop_failed"] = True
            else:
                self._total_bins = max(result["total"], 1)
                ep_hits = set(result["hits"])
                new_hits = ep_hits - self._cum_hits
                self._cum_hits |= ep_hits
                reward = float(len(new_hits)) * 10.0
                if len(new_hits) == 0:
                    reward = -1.0
                info.update({
                    "ep_covered"  : result["covered"],
                    "ep_pct"      : result["pct"],
                    "new_hits"    : len(new_hits),
                    "cum_covered" : len(self._cum_hits),
                    "cum_pct"     : 100.0 * len(self._cum_hits) / self._total_bins,
                    "total_bins"  : self._total_bins,
                })
            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info
