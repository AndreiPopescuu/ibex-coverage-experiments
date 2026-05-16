"""train_l9_hybrid_ppo.py — PPO meta-controller + constrained random generators.

PPO alege la fiecare pas UN MOD (Discrete 6), iar un generator constrained
random produce instructiunea concreta. PPO nu trebuie sa invete mecanica
RISC-V — invata STRATEGIA: cand sa forteze RAW, cand corner, etc.

Moduri:
  0 FREE          — constrained random default (diversitate op + extreme reg)
  1 RAW_PRODUCER  — R-type cu rd != x0 (producator pentru RAW hazard)
  2 RAW_CONSUMER  — consumer cu rs1/rs2 = un rd recent din program
  3 FORCE_CORNER  — imm extrem + registri extremi (x0, x16=-1, x17=INT_MAX, x18=INT_MIN)
  4 FORCE_RDZERO  — rd = x0 (rdzero bins)
  5 FORCE_SAMESRC — rs1 == rs2 (samesrc bins)

De ce e mai bun decat PPO plain:
  - Action space: Discrete(6) vs MultiDiscrete([70,32,32,32,5]) — de ~600k ori mai mic
  - PPO vede direct legatura: mode RAW_CONSUMER → reward din raw bins
  - Constrained random garanteaza calitatea instructiunilor individuale

Target: >62.88% (constrained random 2000 ep Verilator)
"""

import argparse, sys, time, random
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
)
from coverage_model import INSTR_NAMES, INSTR_INDEX, HAS_RD, HAS_RS1, HAS_RS2, N_INSTRS
from codec_l8 import N_OPS, IMM_BUCKETS

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


EPISODE_STEPS  = 480
TOTAL_EPISODES = 2000
N_MODES        = 6
INIT_ENT       = 0.20   # mai mare — action space mic, explorare critica
FINAL_ENT      = 0.02

_CAT_WEIGHT = {
    "seen": 1.0, "opair": 1.5, "raw": 3.0, "seq": 2.5,
    "corner": 2.0, "rdzero": 1.5, "rs1zero": 1.5, "imm": 1.5, "samesrc": 2.0,
}
BASE_REWARD = 10.0


# ── Distributii pentru generatoare ───────────────────────────────────────────

_BASE_OP_PROBS = np.ones(N_OPS, dtype=np.float64)
for _n in ["ADD","SUB","SLL","SLT","SLTU","XOR","SRL","SRA","OR","AND",
           "MUL","DIV","REM","REMU","MULH","MULHSU","MULHU","DIVU"]:
    if _n in INSTR_INDEX and INSTR_INDEX[_n] < N_OPS:
        _BASE_OP_PROBS[INSTR_INDEX[_n]] *= 2.0
for _n in ["BEQ","BNE","BLT","BGE","BLTU","BGEU"]:
    if _n in INSTR_INDEX and INSTR_INDEX[_n] < N_OPS:
        _BASE_OP_PROBS[INSTR_INDEX[_n]] *= 2.5
for _n in ["LW","LH","LHU","SW","SH","JAL","JALR"]:
    if _n in INSTR_INDEX and INSTR_INDEX[_n] < N_OPS:
        _BASE_OP_PROBS[INSTR_INDEX[_n]] *= 1.5
_BASE_OP_PROBS /= _BASE_OP_PROBS.sum()

_REG_PROBS = np.ones(32, dtype=np.float64)
_REG_PROBS[0]  *= 3.0
_REG_PROBS[16] *= 2.0
_REG_PROBS[17] *= 2.0
_REG_PROBS[18] *= 2.0
_REG_PROBS /= _REG_PROBS.sum()

_IMM_PROBS = np.array([3.0, 1.0, 1.0, 1.0, 3.0])
_IMM_PROBS /= _IMM_PROBS.sum()

_R_TYPE = [INSTR_INDEX[n] for n in
           ["ADD","SUB","XOR","OR","AND","SLL","SRL","SRA","SLT","SLTU",
            "MUL","DIV","REM","REMU","MULH","MULHU","MULHSU","DIVU"]
           if n in INSTR_INDEX and INSTR_INDEX[n] < N_OPS]

_CONSUMER_OPS = [INSTR_INDEX[n] for n in
                 ["ADD","SUB","XOR","OR","AND","SLL","SRL","SRA","ADDI","ORI","ANDI",
                  "LW","LH","SW","SH","BEQ","BNE","BLT","BGE","MUL","DIV"]
                 if n in INSTR_INDEX and INSTR_INDEX[n] < N_OPS]

_EXTREME_REGS = [0, 16, 17, 18]   # x0=0, x16=-1, x17=INT_MAX, x18=INT_MIN


# ── Generatoare per mod ───────────────────────────────────────────────────────

def _rand_op(last_op: int) -> int:
    probs = _BASE_OP_PROBS.copy()
    if last_op >= 0:
        probs[last_op] *= 0.1
        probs /= probs.sum()
    return int(np.random.choice(N_OPS, p=probs))


def gen_free(last_op: int, recent_rds: list) -> tuple:
    op_i = _rand_op(last_op)
    rd   = int(np.random.choice(32, p=_REG_PROBS))
    rs1  = int(np.random.choice(32, p=_REG_PROBS))
    rs2  = int(np.random.choice(32, p=_REG_PROBS))
    imm  = int(np.random.choice(IMM_BUCKETS, p=_IMM_PROBS))
    if random.random() < 0.10:
        rs2 = rs1   # samesrc din cand in cand
    return (op_i, rd, rs1, rs2, imm)


def gen_raw_producer(last_op: int, recent_rds: list) -> tuple:
    op_i = random.choice(_R_TYPE)
    rd   = random.randint(1, 31)
    rs1  = int(np.random.choice(32, p=_REG_PROBS))
    rs2  = int(np.random.choice(32, p=_REG_PROBS))
    imm  = int(np.random.choice(IMM_BUCKETS, p=_IMM_PROBS))
    return (op_i, rd, rs1, rs2, imm)


def gen_raw_consumer(last_op: int, recent_rds: list) -> tuple:
    if not recent_rds:
        return gen_free(last_op, recent_rds)
    target_rd = random.choice(recent_rds)
    op_i = random.choice(_CONSUMER_OPS)
    rd   = int(np.random.choice(32, p=_REG_PROBS))
    if random.random() < 0.5:
        rs1, rs2 = target_rd, int(np.random.choice(32, p=_REG_PROBS))
    else:
        rs1, rs2 = int(np.random.choice(32, p=_REG_PROBS)), target_rd
    imm = int(np.random.choice(IMM_BUCKETS, p=_IMM_PROBS))
    return (op_i, rd, rs1, rs2, imm)


def gen_corner(last_op: int, recent_rds: list) -> tuple:
    op_i = _rand_op(last_op)
    rd   = int(np.random.choice(32, p=_REG_PROBS))
    rs1  = random.choice(_EXTREME_REGS)
    rs2  = random.choice(_EXTREME_REGS)
    imm  = random.choice([0, IMM_BUCKETS - 1])   # bucket extrem (zero sau max)
    return (op_i, rd, rs1, rs2, imm)


def gen_rdzero(last_op: int, recent_rds: list) -> tuple:
    op_i = _rand_op(last_op)
    rs1  = int(np.random.choice(32, p=_REG_PROBS))
    rs2  = int(np.random.choice(32, p=_REG_PROBS))
    imm  = int(np.random.choice(IMM_BUCKETS, p=_IMM_PROBS))
    return (op_i, 0, rs1, rs2, imm)


def gen_samesrc(last_op: int, recent_rds: list) -> tuple:
    op_i = _rand_op(last_op)
    rd   = int(np.random.choice(32, p=_REG_PROBS))
    rs   = int(np.random.choice(32, p=_REG_PROBS))
    imm  = int(np.random.choice(IMM_BUCKETS, p=_IMM_PROBS))
    return (op_i, rd, rs, rs, imm)


_GENERATORS = [gen_free, gen_raw_producer, gen_raw_consumer,
               gen_corner, gen_rdzero, gen_samesrc]


# ── Helpers reward ────────────────────────────────────────────────────────────

def _cat_of(hit: str) -> str:
    if hit.startswith("seen_"):    return "seen"
    if hit.startswith("opair_"):   return "opair"
    if hit.startswith("raw"):      return "raw"
    if hit.startswith("seq_"):     return "seq"
    if hit.startswith("corner_"):  return "corner"
    if hit.startswith("rdzero_"):  return "rdzero"
    if hit.startswith("rs1zero_"): return "rs1zero"
    if hit.startswith("imm_"):     return "imm"
    if hit.startswith("samesrc_"): return "samesrc"
    return "seen"


# ── Gym environment ───────────────────────────────────────────────────────────

class IbexL9HybridEnv(gym.Env):
    """PPO alege modul; generatorul produce instructiunea concreta.

    Obs (13-dim):
      [0]  step_frac
      [1]  cum_cov_frac
      [2]  seen_frac
      [3]  opair_frac
      [4]  raw_frac
      [5]  seq_frac
      [6]  corner_frac
      [7]  rdzero_frac
      [8]  rs1zero_frac
      [9]  imm_frac
      [10] samesrc_frac
      [11] ep_idx_norm
      [12] has_recent_rd  — 1.0 daca exista un rd recent disponibil pentru consumer
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_steps=480, max_episodes=2000, driver="test_run_for_l9"):
        super().__init__()
        self.episode_steps = episode_steps
        self.max_episodes  = max_episodes
        self.driver        = driver

        self.action_space      = spaces.Discrete(N_MODES)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # Stare cumulata
        self._cum_hits = set()
        self._cat = {k: 0 for k in _CAT_WEIGHT}
        self._n_episodes = 0

        # Stare per-episod
        self._actions    : list = []
        self._step_idx   : int  = 0
        self._last_op    : int  = -1
        self._recent_rds : list = []   # ultimii 3 rd scrisi

    def _obs(self) -> np.ndarray:
        return np.array([
            self._step_idx / max(self.episode_steps, 1),               # [0]
            min(1.0, len(self._cum_hits) / max(_TOTAL_BINS, 1)),       # [1]
            min(1.0, self._cat["seen"]    / max(_SEEN_T,    1)),       # [2]
            min(1.0, self._cat["opair"]   / max(_OPAIR_T,  1)),       # [3]
            min(1.0, self._cat["raw"]     / max(_RAW_T,    1)),       # [4]
            min(1.0, self._cat["seq"]     / max(_SEQ_T,    1)),       # [5]
            min(1.0, self._cat["corner"]  / max(_CORNER_T, 1)),       # [6]
            min(1.0, self._cat["rdzero"]  / max(_RDZERO_T, 1)),       # [7]
            min(1.0, self._cat["rs1zero"] / max(_RS1ZERO_T,1)),       # [8]
            min(1.0, self._cat["imm"]     / max(_IMM_T,    1)),       # [9]
            min(1.0, self._cat["samesrc"] / max(_SAMESRC_T,1)),       # [10]
            min(1.0, self._n_episodes / max(self.max_episodes, 1)),   # [11]
            1.0 if self._recent_rds else 0.0,                         # [12]
        ], dtype=np.float32)

    def _cat_saturation(self, cat: str) -> float:
        totals = {"seen": _SEEN_T, "opair": _OPAIR_T, "raw": _RAW_T, "seq": _SEQ_T,
                  "corner": _CORNER_T, "rdzero": _RDZERO_T, "rs1zero": _RS1ZERO_T,
                  "imm": _IMM_T, "samesrc": _SAMESRC_T}
        t = totals.get(cat, 1)
        return 1.0 - min(1.0, self._cat[cat] / max(t, 1))

    def _shaped_reward(self, new_hits: set) -> float:
        if not new_hits:
            return -1.0
        total = 0.0
        for h in new_hits:
            cat   = _cat_of(h)
            w     = _CAT_WEIGHT.get(cat, 1.0)
            sat   = self._cat_saturation(cat)
            total += BASE_REWARD * w * (1.0 + sat)
        return total

    def reset(self, *, seed=None, options=None):
        self._actions.clear()
        self._step_idx   = 0
        self._last_op    = -1
        self._recent_rds.clear()
        return self._obs(), {}

    def step(self, action):
        mode = int(action)
        instr = _GENERATORS[mode](self._last_op, list(self._recent_rds))
        op_i, rd, rs1, rs2, imm_b = instr
        self._actions.append(instr)
        self._step_idx += 1

        # Actualizeaza starea locala pentru obs si generatoare
        self._last_op = op_i
        if rd != 0:
            self._recent_rds.append(rd)
            if len(self._recent_rds) > 3:
                self._recent_rds.pop(0)

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

                reward = self._shaped_reward(new_hits)

                for h in new_hits:
                    self._cat[_cat_of(h)] += 1
                self._cum_hits |= ep_hits

                cum_pct = 100.0 * len(self._cum_hits) / max(_TOTAL_BINS, 1)
                info.update({
                    "ep_pct"  : result["pct"],
                    "cum_pct" : cum_pct,
                    "new_hits": len(new_hits),
                })

            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info


# ── Callback ──────────────────────────────────────────────────────────────────

class CoverageCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.last_info: dict = {}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", self.locals.get("done"))
        infos = self.locals.get("infos", self.locals.get("info"))
        if dones is None or infos is None:
            return True
        done_arr = np.asarray(dones).flatten()
        info_arr = [infos] if isinstance(infos, dict) else list(infos)
        for i, done in enumerate(done_arr):
            if bool(done) and i < len(info_arr) and info_arr[i]:
                self.last_info = info_arr[i]
        return True


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",  type=int, default=TOTAL_EPISODES)
    ap.add_argument("--steps",     type=int, default=EPISODE_STEPS)
    ap.add_argument("--driver",    default="test_run_for_l9")
    ap.add_argument("--sim-build", default=None)
    ap.add_argument("--out",       default="l9_hybrid_ppo")
    args = ap.parse_args()

    if args.sim_build:
        import os
        os.environ["IBEX_SIM_BUILD"] = args.sim_build

    env = IbexL9HybridEnv(
        episode_steps=args.steps,
        max_episodes=args.episodes,
        driver=args.driver,
    )

    print("=" * 68)
    print("PPO HYBRID L9 — meta-controller (Discrete 6) + constrained generators")
    print(f"  {args.steps} pasi/ep  x  {args.episodes} ep")
    print(f"  Obs: 13-dim  |  Action: Discrete({N_MODES})")
    print(f"  ent_coef: {INIT_ENT} → {FINAL_ENT}  |  Category weights: raw={_CAT_WEIGHT['raw']}x")
    print(f"  Total bins: {_TOTAL_BINS}")
    print(f"  Target: >62.88% (constrained random 2000 ep)")
    print(f"  Estimat: {args.episodes * 7 / 3600:.1f} ore la ~7s/ep")
    print("=" * 68)

    model = PPO(
        "MlpPolicy", env,
        n_steps       = args.steps,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        learning_rate = 3e-4,
        ent_coef      = INIT_ENT,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 0,
        device        = "cpu",
        policy_kwargs = dict(net_arch=[128, 128]),   # retea mica — problema simpla
    )

    callback     = CoverageCallback()
    ep_pcts      = []
    cum_pcts     = []
    new_hits_log = []
    mode_counts  = np.zeros(N_MODES, dtype=int)
    t_total      = time.time()

    print(f"\n{'ep':>6}  {'ep%':>7}  {'cum%':>8}  {'new':>6}  {'ent':>6}  {'s':>5}")
    print("-" * 52)

    for ep in range(1, args.episodes + 1):
        t0 = time.time()

        frac = (ep - 1) / max(args.episodes - 1, 1)
        model.ent_coef = INIT_ENT + frac * (FINAL_ENT - INIT_ENT)

        model.learn(
            total_timesteps     = args.steps,
            callback            = callback,
            reset_num_timesteps = (ep == 1),
        )

        info    = callback.last_info
        ep_pct  = info.get("ep_pct",   0.0)
        cum_pct = info.get("cum_pct",  0.0)
        new_n   = info.get("new_hits", 0)
        elapsed = time.time() - t0

        ep_pcts.append(ep_pct)
        cum_pcts.append(cum_pct)
        new_hits_log.append(new_n)

        if ep % 10 == 0 or ep == 1:
            print(f"{ep:>6}  {ep_pct:>6.2f}%  {cum_pct:>7.2f}%  "
                  f"{new_n:>6}  {model.ent_coef:>5.3f}  {elapsed:>4.1f}s")

        if ep % 100 == 0:
            avg_new = sum(new_hits_log[-100:]) / 100
            ckpt = THIS / f"{args.out}_checkpoint_ep{ep}"
            model.save(str(ckpt))
            eta_h = (args.episodes - ep) * 7 / 3600
            print(f"  ── ep {ep}: cum={cum_pct:.2f}%  avg_new={avg_new:.1f}  "
                  f"[checkpoint]  ETA: {eta_h:.1f}h ──")

    print("=" * 68)
    print(f"Final cum coverage: {cum_pcts[-1]:.2f}%")
    print(f"Constrained random baseline: 62.88%  →  "
          f"{'BATUT! ✓' if cum_pcts[-1] > 62.88 else 'sub baseline'}")
    print(f"Wall time: {time.time() - t_total:.0f}s")

    np.savez(
        str(THIS / f"{args.out}_curve.npz"),
        ep_pct   = np.array(ep_pcts),
        cum_pct  = np.array(cum_pcts),
        new_hits = np.array(new_hits_log),
    )
    model.save(str(THIS / f"{args.out}_final"))
    print(f"[saved] {args.out}_curve.npz + {args.out}_final.zip")


if __name__ == "__main__":
    main()
