"""train_l8.py v2 — PPO cu hyperparametri tunați să bată random.

Schimbări față de v1:
  - n_steps = episode_steps = 128  (rollout = 1 episod complet)
  - ent_coef = 0.15 → 0.02         (annealing: explorare inițial, exploatare tardiv)
  - batch_size = 64                 (potrivit pentru episoade scurte)
  - net_arch = [512, 512, 256]      (mai adâncă pentru obs 72-dim)
  - learning_rate cu schedule       (descreștere liniară)
  - Rulează mai întâi directed (seed de 420 linii gratuite)

Usage:
    python train_l8.py                    # 500 episoade
    python train_l8.py --episodes 200     # mai puțin
    python train_l8.py --no-directed      # sari peste directed
    python train_l8.py --resume           # continuă din checkpoint
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l8 import IbexL8Env
from directed_l8 import run_directed

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3")
    sys.exit(1)


class CovCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.ep_line_pcts  = []
        self.cum_line_pcts = []
        self.new_hits_list = []
        self.rewards       = []
        self.diversities   = []
        self._ep_rew = 0.0

    def _on_step(self):
        self._ep_rew += float(self.locals["rewards"][0])
        info = (self.locals.get("infos") or [{}])[0]
        if self.locals.get("dones", [False])[0]:
            self.ep_line_pcts.append(info.get("ep_line_pct", 0.0))
            self.cum_line_pcts.append(info.get("cum_line_pct", 0.0))
            self.new_hits_list.append(info.get("new_line_hits", 0))
            self.rewards.append(self._ep_rew)
            self.diversities.append(info.get("op_diversity", 0))
            self._ep_rew = 0.0
            ep = len(self.cum_line_pcts)
            if self.verbose and ep % 10 == 0:
                print(
                    f"  ep={ep:4d} | "
                    f"line_ep={self.ep_line_pcts[-1]:.1f}% | "
                    f"line_cum={self.cum_line_pcts[-1]:.1f}% | "
                    f"new={self.new_hits_list[-1]:4d} | "
                    f"diversity={self.diversities[-1]:2d} ops | "
                    f"rew={self.rewards[-1]:.0f}"
                )
        return True

    def save(self, path):
        np.savez(path,
            ep_line_pct  = np.array(self.ep_line_pcts),
            cum_line_pct = np.array(self.cum_line_pcts),
            new_hits     = np.array(self.new_hits_list),
            rewards      = np.array(self.rewards),
            diversities  = np.array(self.diversities),
        )
        print(f"[saved] {path}.npz")


def linear_schedule(initial: float, final: float, total_steps: int):
    """Annealing liniar de la initial la final pe parcursul antrenamentului."""
    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1.0 la start → 0.0 la final
        return final + progress_remaining * (initial - final)
    return schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",    type=int, default=10)
    ap.add_argument("--steps",       type=int, default=128, help="instrucțiuni/episod")
    ap.add_argument("--no-directed", action="store_true")
    ap.add_argument("--resume",      action="store_true")
    ap.add_argument("--eval-only",   action="store_true")
    ap.add_argument("--driver",      default="test_run_for_l8")
    args = ap.parse_args()

    env = IbexL8Env(episode_steps=args.steps, driver=args.driver)

    # ── Step 1: directed seed ─────────────────────────────────────────────
    if not args.no_directed and not args.eval_only:
        print("=" * 60)
        print("STEP 1: Directed seed run")
        print("=" * 60)
        run_directed()
        print()

    model_path = THIS / "l8_ppo_v2"

    # ── Step 2: PPO ───────────────────────────────────────────────────────
    total_steps = args.episodes * args.steps

    ppo_kwargs = dict(
        # Rollout = exact 1 episod complet
        n_steps     = args.steps,
        batch_size  = 64,
        n_epochs    = 8,
        gamma       = 0.99,
        gae_lambda  = 0.95,
        clip_range  = 0.2,
        # Learning rate: 3e-4 → 1e-5 (annealing liniar)
        learning_rate = linear_schedule(3e-4, 1e-5, total_steps),
        # Entropy: 0.15 → 0.01 (explorare → exploatare)
        ent_coef    = linear_schedule(0.15, 0.01, total_steps),
        vf_coef     = 0.5,
        max_grad_norm = 0.5,
        verbose     = 0,
        policy_kwargs = dict(
            net_arch = [512, 512, 256],  # rețea mai adâncă pentru obs 72-dim
        ),
    )

    if args.eval_only:
        model = PPO.load(str(model_path), env=env)
    elif args.resume and model_path.with_suffix(".zip").exists():
        print(f"Resuming from {model_path}.zip")
        model = PPO.load(str(model_path), env=env)
    else:
        print("=" * 60)
        print(f"STEP 2: PPO v2 — {args.episodes} episoade × {args.steps} pași")
        print(f"  obs_dim={2 + 70}, reward=novelty×100")
        print(f"  ent_coef: 0.15→0.01, lr: 3e-4→1e-5")
        print("=" * 60)
        model = PPO("MlpPolicy", env, **ppo_kwargs)

    callback = CovCallback(verbose=1)

    if not args.eval_only:
        t0 = time.time()
        model.learn(
            total_timesteps=total_steps,
            callback=callback,
            reset_num_timesteps=not args.resume,
        )
        elapsed = time.time() - t0
        print(f"\nAntrenament gata: {elapsed:.0f}s  ({elapsed/args.episodes:.1f}s/ep)")
        model.save(str(model_path))
        callback.save(str(THIS / "l8_ppo_v2_curve"))

    # ── Evaluare finală ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUARE FINALĂ (10 episoade, policy deterministic)")
    print("=" * 60)
    ep_pcts = []
    for i in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, done, info = env.step(action)
        ep_pcts.append(info.get("ep_line_pct", 0.0))
        cum = info.get("cum_line_pct", 0.0)
        new = info.get("new_line_hits", 0)
        div = info.get("op_diversity", 0)
        print(f"  ep {i+1:2d}: line_ep={ep_pcts[-1]:.1f}%  cum={cum:.1f}%  new={new}  diversity={div}")

    print(f"\nMean ep line coverage : {np.mean(ep_pcts):.1f}%")
    print(f"Cumulative line coverage: {100*len(env._cum_line_hits)/max(env._line_total,1):.1f}%")

if __name__ == "__main__":
    main()
