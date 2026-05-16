"""train_l9_finetune.py — Shadow → RTL fine-tuning cu reward shaping imbunatatit.

Strategii:
  1. Transfer learning: incarca l9_shadow_final.zip, fine-tune pe Verilator cu LR mic
  2. Category weights: raw/seq/corner primesc multiplicator mai mare (sunt mai greu de acoperit)
  3. Saturation bonus: bins din categorii putin acoperite primesc bonus suplimentar
  4. Ent_coef curriculum: 0.10 → 0.02 pe parcursul training-ului
  5. Un singur run Verilator per episod (callback in loc de predict loop suplimentar)

Target: >62.88% (constrained random cu 2000 ep Verilator)

Usage:
    python train_l9_finetune.py
    python train_l9_finetune.py --episodes 1500 --shadow l9_shadow_final
    python train_l9_finetune.py --from-scratch   # fara transfer, pentru comparatie
"""

import argparse, sys, time, math
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
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3 torch"); sys.exit(1)


EPISODE_STEPS  = 480
TOTAL_EPISODES = 1500
INIT_LR        = 5e-5    # mai mic decat 2e-4 din scratch (fine-tuning)
INIT_ENT       = 0.10
FINAL_ENT      = 0.02

# Multiplicatori per categorie — bins greu de generat primesc bonus mai mare
_CAT_WEIGHT = {
    "seen":    1.0,
    "opair":   1.5,
    "raw":     3.0,   # RAW hazards necesita secvente producer→consumer
    "seq":     2.5,   # perechi consecutive de instructiuni
    "corner":  2.0,   # valori de colt (INT_MIN, 0, -1 etc.)
    "rdzero":  1.5,
    "rs1zero": 1.5,
    "imm":     1.5,
    "samesrc": 1.5,
}
BASE_REWARD = 10.0


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


class IbexL9FinetuneEnv(gym.Env):
    """RTL env cu category-weighted reward pentru fine-tuning din shadow."""

    metadata = {"render_modes": []}

    def __init__(self, episode_steps=480, max_episodes=1500, driver="test_run_for_l9"):
        super().__init__()
        self.episode_steps = episode_steps
        self.max_episodes  = max_episodes
        self.driver        = driver

        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )

        # Stare cumulata (persista intre episoade)
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

    # ── Helpers obs ──────────────────────────────────────────────────────────

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
            self._step_idx / max(self.episode_steps, 1),               # [0] step_frac
            min(1.0, len(self._cum_hits) / _TOTAL_BINS),               # [1] cum_cov_frac
            min(1.0, self._cat["seen"]   / max(_SEEN_T,    1)),        # [2] seen_frac
            min(1.0, self._cat["opair"]  / max(_OPAIR_T,   1)),        # [3] opair_frac
            min(1.0, self._cat["raw"]    / max(_RAW_T,     1)),        # [4] raw_frac
            min(1.0, self._cat["seq"]    / max(_SEQ_T,     1)),        # [5] seq_frac
            min(1.0, self._cat["corner"] / max(_CORNER_T,  1)),        # [6] corner_frac
            0.0,                                                         # [7] ep_new (necunoscut mid-episode)
            self._last_op / max(N_OPS - 1, 1),                         # [8] last_op_norm
            min(1.0, self._n_episodes / max(self.max_episodes, 1)),    # [9] ep_idx_norm
            self._last_rd / 31.0,                                      # [10] last_rd_norm
            float(len(self._recent_rds)) / 3.0,                        # [11] recent_rds_frac
            self._seq_frac_cache[op_i] if op_i < N_OPS else 0.0,      # [12] seq_covered_last_op
            self._raw_frac_cache[op_i] if op_i < N_OPS else 0.0,      # [13] raw_uncov_consumer
            min(1.0, self._cat["rdzero"]  / max(_RDZERO_T,  1)),       # [14] rdzero_frac
            min(1.0, self._cat["rs1zero"] / max(_RS1ZERO_T, 1)),       # [15] rs1zero_frac
            min(1.0, self._cat["imm"]     / max(_IMM_T,     1)),       # [16] imm_frac
            min(1.0, self._cat["samesrc"] / max(_SAMESRC_T, 1)),       # [17] samesrc_frac
        ], dtype=np.float32)

    def _cat_saturation(self, cat: str) -> float:
        """Fractia NEACOPERITA din categorie — 1.0=virgin, 0.0=complet acoperita."""
        totals = {"seen": _SEEN_T, "opair": _OPAIR_T, "raw": _RAW_T, "seq": _SEQ_T,
                  "corner": _CORNER_T, "rdzero": _RDZERO_T, "rs1zero": _RS1ZERO_T,
                  "imm": _IMM_T, "samesrc": _SAMESRC_T}
        t = totals.get(cat, 1)
        return 1.0 - min(1.0, self._cat[cat] / max(t, 1))

    def _shaped_reward(self, new_hits: set) -> float:
        """Reward cu category weights + saturation bonus.

        Formula per bin nou:
          reward += BASE * weight * (1 + saturation_bonus)
        unde saturation_bonus = fractia categoriei inca neacoperita (0→1).
        Bins dintr-o categorie virgin primesc +100% bonus; cele din categorie
        aproape saturata nu primesc bonus suplimentar.
        """
        if not new_hits:
            return -1.0
        total = 0.0
        for h in new_hits:
            cat     = _cat_of(h)
            weight  = _CAT_WEIGHT.get(cat, 1.0)
            sat_b   = self._cat_saturation(cat)   # [0,1], mai mare = categoria mai goala
            total  += BASE_REWARD * weight * (1.0 + sat_b)
        return total

    def _update_state(self, new_hits: set):
        for h in new_hits:
            cat = _cat_of(h)
            self._cat[cat] += 1
            pair = _SEQ_NAME_TO_PAIR.get(h)
            if pair:
                self._cum_seq_pairs.add(pair)
            triple = _RAW_NAME_TO_TRIPLE.get(h)
            if triple:
                self._cum_raw_pairs.add(triple)

    # ── Gym interface ─────────────────────────────────────────────────────────

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

                reward = self._shaped_reward(new_hits)

                # Actualizeaza starea DUPA calcul reward (saturation_bonus foloseste starea veche)
                self._update_state(new_hits)
                self._cum_hits |= ep_hits

                cum_pct = 100.0 * len(self._cum_hits) / _TOTAL_BINS
                info.update({
                    "ep_pct"  : result["pct"],
                    "cum_pct" : cum_pct,
                    "new_hits": len(new_hits),
                })

                for i in range(min(N_OPS, N_INSTRS)):
                    self._seq_frac_cache[i] = self._seq_covered_frac(i)
                    self._raw_frac_cache[i] = self._raw_uncov_consumer_frac(i)

            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info


# ── Callback pentru logging fara al doilea run Verilator ─────────────────────

class CoverageCallback(BaseCallback):
    """Captureaza info din ultimul step al episodului fara a rula inca un episod."""

    def __init__(self):
        super().__init__()
        self.last_info: dict = {}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", self.locals.get("done"))
        infos = self.locals.get("infos", self.locals.get("info"))
        if dones is None or infos is None:
            return True
        import numpy as np
        done_arr = np.asarray(dones).flatten()
        info_arr = [infos] if isinstance(infos, dict) else list(infos)
        for i, done in enumerate(done_arr):
            if bool(done) and i < len(info_arr) and info_arr[i]:
                self.last_info = info_arr[i]
        return True


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",     type=int,   default=TOTAL_EPISODES)
    ap.add_argument("--steps",        type=int,   default=EPISODE_STEPS)
    ap.add_argument("--shadow",       default="l9_shadow_final",
                    help="Checkpoint shadow de incarcat (fara .zip)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Antreneaza de la zero (fara transfer), pentru comparatie")
    ap.add_argument("--driver",       default="test_run_for_l9")
    ap.add_argument("--sim-build",    default=None)
    ap.add_argument("--out",          default="l9_finetune",
                    help="Prefix pentru fisierele de output")
    args = ap.parse_args()

    if args.sim_build:
        import os
        os.environ["IBEX_SIM_BUILD"] = args.sim_build

    env = IbexL9FinetuneEnv(
        episode_steps=args.steps,
        max_episodes=args.episodes,
        driver=args.driver,
    )

    print("=" * 68)
    if args.from_scratch:
        print("PPO FINE-TUNE L9 — ANTRENAT DE LA ZERO (comparatie)")
    else:
        print(f"PPO FINE-TUNE L9 — TRANSFER DIN {args.shadow}.zip")
    print(f"  {args.steps} pasi/ep  x  {args.episodes} ep")
    print(f"  LR initial: {INIT_LR:.0e}  |  ent_coef: {INIT_ENT} → {FINAL_ENT}")
    print(f"  Category weights: raw={_CAT_WEIGHT['raw']}x  seq={_CAT_WEIGHT['seq']}x  "
          f"corner={_CAT_WEIGHT['corner']}x")
    print(f"  Total bins: {_TOTAL_BINS}")
    print(f"  Target de batut: constrained random 62.88% (2000 ep)")
    print(f"  Estimat: {args.episodes * 7 / 3600:.1f} ore la ~7s/ep")
    print("=" * 68)

    shadow_path = THIS / f"{args.shadow}.zip"

    # Construim intotdeauna modelul nou cu obs 18-dim
    model = PPO(
        "MlpPolicy", env,
        n_steps       = args.steps,
        batch_size    = 96,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        learning_rate = INIT_LR,
        ent_coef      = INIT_ENT,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 0,
        device        = "cpu",
        policy_kwargs = dict(net_arch=[512, 512, 256]),
    )

    if not args.from_scratch and shadow_path.exists():
        print(f"\n[transfer] {shadow_path}  (obs 14→18)")
        shadow = PPO.load(str(THIS / args.shadow), device="cpu")
        src = dict(shadow.policy.named_parameters())
        with th.no_grad():
            for name, param in model.policy.named_parameters():
                if name not in src:
                    continue
                s = src[name]
                if s.shape == param.shape:
                    param.copy_(s)
                elif param.shape[1] == 18 and s.shape[1] == 14:
                    # Input layer: copiaza primele 14 coloane, init 0 pentru cele 4 noi
                    param[:, :14] = s
                    param[:, 14:] = 0.0
                    print(f"  expand input: {name} {list(s.shape)} → {list(param.shape)}")
        del shadow
        print(f"[transfer] OK — LR={INIT_LR:.0e}, ent={INIT_ENT}")
    else:
        if not args.from_scratch:
            print(f"[warn] {shadow_path} nu exista — antrenez de la zero")

    callback  = CoverageCallback()
    ep_pcts   = []
    cum_pcts  = []
    new_hits_log = []
    t_total   = time.time()

    print(f"\n{'ep':>6}  {'ep%':>7}  {'cum%':>8}  {'new':>6}  {'ent':>6}  {'s':>5}")
    print("-" * 52)

    for ep in range(1, args.episodes + 1):
        t0 = time.time()

        # Curriculum ent_coef: decrestere liniara
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
