"""env_l8.py v2 — PPO-optimized for line coverage.

Schimbări față de v1:
  - episode_steps = 128  (față de 1024 — feedback de 8x mai des)
  - Observație 72-dim: [step_frac, cum_line_frac] + histogramă 70 ops normalizată
    → PPO vede ce instrucțiuni a folosit deja și evită redundanța
  - Reward = pur novelty: DOAR linii noi față de cumulat × 100
    → zero reward pentru linii deja cunoscute, forțează explorare continuă
  - Penalitate ușoară dacă nu găsește nimic nou (−1)
    → descurajează secvențele sterile
"""

import os, sys, json, subprocess
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS = Path(__file__).resolve().parent
L5   = (THIS.parent / "level5_real_rtl").resolve()
sys.path.insert(0, str(L5))
import cov_parser

sys.path.insert(0, str(THIS))
from codec_l8 import N_OPS, IMM_BUCKETS, emit_program

ML4DV    = (THIS.parent.parent / "cpu").resolve()
_SIM_BUILD = os.environ.get("IBEX_SIM_BUILD", "sim_build")
VTOP     = ML4DV / _SIM_BUILD / "Vtop"
COVDAT   = ML4DV / "coverage.dat"
PROG_JSON = "/tmp/rl_l8_program.json"


def run_program(actions, driver="test_run_for_l8"):
    machine = emit_program(actions)
    with open(PROG_JSON, "w") as f:
        json.dump({"n": len(machine), "agent": "l8v2",
                   "machine_code": [int(m) for m in machine]}, f)
    import sysconfig as _sc
    _pylib = _sc.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu"
        + ((":" + _pylib) if _pylib else "")
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["MODULE"]     = driver
    env["RL_L8_JSON"] = PROG_JSON
    proc = subprocess.run(
        [str(VTOP)], cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


class IbexL8Env(gym.Env):
    """
    Action space : MultiDiscrete([N_OPS=70, 32, 32, 32, IMM_BUCKETS=5])
    Obs space    : Box(72,)
        [0]   step / episode_steps          (progres în episod)
        [1]   cum_line_covered / line_total  (coverage cumulat)
        [2..71] op_counts[i] / episode_steps (câte instrucțiuni de tip i am emis)

    Reward : pur novelty
        +100 per linie nouă față de cumulat
        -1   dacă episodul nu aduce nicio linie nouă (descurajează secvențe sterile)
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_steps: int = 128,
        driver: str = "test_run_for_l8",
    ):
        super().__init__()
        self.episode_steps = episode_steps
        self.driver        = driver

        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        # 2 scalari + histogramă 70 ops
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 + N_OPS,), dtype=np.float32
        )

        self._actions    : list[tuple] = []
        self._op_counts  : np.ndarray = np.zeros(N_OPS, dtype=np.float32)
        self._step_idx   : int = 0
        self._n_episodes : int = 0

        self._cum_line_hits : set[str] = set()
        self._line_total    : int = 1

    def _obs(self) -> np.ndarray:
        obs = np.empty(2 + N_OPS, dtype=np.float32)
        obs[0] = self._step_idx / self.episode_steps
        obs[1] = min(1.0, len(self._cum_line_hits) / max(self._line_total, 1))
        obs[2:] = self._op_counts / max(self._step_idx, 1)
        return obs

    def reset(self, *, seed=None, options=None):
        self._actions.clear()
        self._op_counts[:] = 0.0
        self._step_idx = 0
        return self._obs(), {}

    def step(self, action):
        op_i = int(action[0])
        self._actions.append(tuple(int(x) for x in action))
        self._op_counts[op_i] += 1.0
        self._step_idx += 1

        truncated = (self._step_idx >= self.episode_steps)
        reward, info = 0.0, {}

        if truncated:
            summary = run_program(self._actions, driver=self.driver)
            if summary is None:
                reward = -5.0
                info["vtop_failed"] = True
            else:
                line_cov, line_tot = summary.by_kind.get("line", (0, 1))
                self._line_total = max(line_tot, 1)

                line_prefix = "\x01page\x02v_line/"
                ep_line_hits = {
                    k for k, v in summary.points.items()
                    if v > 0 and line_prefix in ("\x01" + k)
                }
                new_hits = ep_line_hits - self._cum_line_hits
                self._cum_line_hits |= ep_line_hits

                # Reward pur novelty
                if len(new_hits) > 0:
                    reward = float(len(new_hits)) * 100.0
                else:
                    reward = -1.0   # penalitate pentru episod steril

                info.update({
                    "ep_line_pct"     : 100.0 * line_cov / self._line_total,
                    "new_line_hits"   : len(new_hits),
                    "cum_line_covered": len(self._cum_line_hits),
                    "cum_line_pct"    : 100.0 * len(self._cum_line_hits) / self._line_total,
                    "branch_pct"      : 100.0 * summary.by_kind.get("branch",(0,1))[0]
                                        / max(summary.by_kind.get("branch",(0,1))[1], 1),
                    "toggle_pct"      : summary.kind_pct("toggle"),
                    "op_diversity"    : int(np.count_nonzero(self._op_counts)),
                })

            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info
