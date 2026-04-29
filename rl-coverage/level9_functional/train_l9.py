"""train_l9.py — PPO training pentru Rich Functional Coverage.

Usage:
    python train_l9.py                    # 300 episoade
    python train_l9.py --episodes 100
    python train_l9.py --no-directed      # sari peste directed seed
    python train_l9.py --resume
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l9 import IbexL9Env

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


class CovCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.ep_pcts   = []
        self.cum_pcts  = []
        self.new_hits  = []
        self._ep_rew   = 0.0

    def _on_step(self):
        self._ep_rew += float(self.locals["rewards"][0])
        info = (self.locals.get("infos") or [{}])[0]
        if self.locals.get("dones", [False])[0]:
            self.ep_pcts.append(info.get("ep_pct", 0.0))
            self.cum_pcts.append(info.get("cum_pct", 0.0))
            self.new_hits.append(info.get("new_hits", 0))
            ep = len(self.cum_pcts)
            if self.verbose and ep % 1 == 0:
                print(f"  ep={ep:4d} | ep={self.ep_pcts[-1]:.1f}% | "
                      f"cum={self.cum_pcts[-1]:.1f}% | "
                      f"new={self.new_hits[-1]} | rew={self._ep_rew:.0f}")
            self._ep_rew = 0.0
        return True

    def save(self, path):
        np.savez(path,
            ep_pct  = np.array(self.ep_pcts),
            cum_pct = np.array(self.cum_pcts),
            new_hits= np.array(self.new_hits),
        )
        print(f"[saved] {path}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",    type=int, default=300)
    ap.add_argument("--steps",       type=int, default=256)
    ap.add_argument("--no-directed", action="store_true")
    ap.add_argument("--resume",      action="store_true")
    ap.add_argument("--eval-only",   action="store_true")
    ap.add_argument("--driver",      default="test_run_for_l9")
    args = ap.parse_args()

    env = IbexL9Env(episode_steps=args.steps, driver=args.driver)

    if not args.no_directed and not args.eval_only:
        print("="*55)
        print("STEP 1: Directed seed (pentru bins de baza)")
        print("="*55)
        try:
            sys.path.insert(0, str(THIS.parent / "level8_directed"))
            from directed_l8 import run_directed
            run_directed()
        except Exception as e:
            print(f"[WARN] directed skip: {e}")

    model_path = THIS / "l9_ppo_model"
    total_steps = args.episodes * args.steps

    ppo_kwargs = dict(
        n_steps       = args.steps,
        batch_size    = 64,
        n_epochs      = 8,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        learning_rate = 3e-4,
        ent_coef      = 0.20,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 0,
        device        = "cpu",
        policy_kwargs = dict(net_arch=[512, 512, 256]),
    )

    if args.eval_only:
        model = PPO.load(str(model_path), env=env)
    elif args.resume and model_path.with_suffix(".zip").exists():
        model = PPO.load(str(model_path), env=env)
    else:
        print("="*55)
        print(f"STEP 2: PPO — {args.episodes} ep × {args.steps} pași")
        print("="*55)
        model = PPO("MlpPolicy", env, **ppo_kwargs)

    callback = CovCallback(verbose=1)
    if not args.eval_only:
        t0 = time.time()
        model.learn(total_timesteps=total_steps, callback=callback,
                    reset_num_timesteps=not args.resume)
        print(f"\nAntrenament: {time.time()-t0:.0f}s")
        model.save(str(model_path))
        callback.save(str(THIS / "l9_ppo_curve"))

    print("\n"+"="*55)
    print("EVALUARE FINALA (10 episoade)")
    print("="*55)
    for i in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, done, info = env.step(action)
        print(f"  ep {i+1:2d}: ep={info.get('ep_pct',0):.1f}%  "
              f"cum={info.get('cum_pct',0):.1f}%  new={info.get('new_hits',0)}")
    print(f"\nFinal cum: {100*len(env._cum_hits)/max(env._total_bins,1):.2f}%")


if __name__ == "__main__":
    main()
