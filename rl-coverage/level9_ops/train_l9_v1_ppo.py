"""train_l9_v1_ppo.py — PPO pe L9 cu obs space mic (10 dims).

Configurație:
  - IbexL9V1Env: 83 ops, obs 10 dims (8 module + step + cum_cov)
  - net_arch: [256, 128]  (simplu, ca baseline)
  - ent_coef: 0.05

Usage:
    python train_l9_v1_ppo.py
    python train_l9_v1_ppo.py --episodes 1200 --steps 256
    python train_l9_v1_ppo.py --episodes 1200 --steps 64
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l9_v1 import IbexL9V1Env, TRACKED_MODULES

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


class Log(BaseCallback):
    def __init__(self, baseline_pct: float):
        super().__init__()
        self.baseline_pct = baseline_pct
        self.history = []

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "cum_pct" not in info:
            return True
        ep      = len(self.history) + 1
        cum     = info["cum_pct"]
        new     = info.get("new_hits_vs_cum", 0)
        worst   = info.get("worst_mod", "?")
        worst_p = info.get("worst_pct", 0.0)
        self.history.append({
            "ep": ep, "cum_pct": cum,
            "ep_pct": info["ep_pct"], "new_hits": new,
        })
        delta = cum - self.baseline_pct
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | cum {cum:>5.2f}% | "
              f"new {new:>4} | Δ {delta:>+6.2f}pp | worst: {worst} {worst_p:.1f}%",
              flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--steps",    type=int, default=256,
                    help="Instrucțiuni per episod")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="l9_v1_ppo_curve.npz")
    args = ap.parse_args()

    print("=" * 64)
    print(f"L9 V1 PPO — obs 10 dims, {args.steps} pași/episod, 83 ops")
    print("=" * 64)

    env = IbexL9V1Env(episode_steps=args.steps, seed=args.seed)

    print(f"  Episoade:    {args.episodes}")
    print(f"  Pași/episod: {args.steps}")
    print(f"  Obs dims:    10")
    print(f"  Action ops:  83  (vs 70 în L8)")
    print(f"  Net arch:    [256, 128]")
    print(f"  ent_coef:    0.05")
    print(f"\n{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δbaseline':>10} | worst module")
    print("-" * 72)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=args.steps,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[256, 128]),
        verbose=0, seed=args.seed, device="cpu",
    )

    cb = Log(baseline_pct=0.0)
    t0 = time.time()
    model.learn(total_timesteps=args.episodes * args.steps, callback=cb)
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed/60:.1f} min")

    if cb.history:
        best  = max(h["cum_pct"] for h in cb.history)
        final = cb.history[-1]["cum_pct"]
        print(f"\nRezultate finale:")
        print(f"  V1 PPO best:  {best:.2f}%")
        print(f"  V1 PPO final: {final:.2f}%")

        eps    = np.array([h["ep"]      for h in cb.history])
        cum    = np.array([h["cum_pct"] for h in cb.history])
        ep_pct = np.array([h["ep_pct"]  for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
