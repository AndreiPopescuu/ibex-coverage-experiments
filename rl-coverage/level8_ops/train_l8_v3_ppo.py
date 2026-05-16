"""train_l8_v3_ppo.py — PPO cu obs space extins (35 dims).

Față de train_l8_dynamic_ppo_v2.py:
  - IbexL8V3Env: 28 module + n_ep_frac + action history (35 dims total)
  - net_arch mai mare: [512, 256]
  - ent_coef crescut la 0.08 (explorare mai agresivă)

Usage:
    python train_l8_v3_ppo.py
    python train_l8_v3_ppo.py --episodes 1200 --steps 256
    python train_l8_v3_ppo.py --episodes 1200 --steps 64
    python train_l8_v3_ppo.py --episodes 1200 --hits l8_pipeline_hits.pkl
"""

import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l8_v3 import IbexL8V3Env, MODULES
from directed_l8 import run_raw

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
                    help="Instrucțiuni per episod (încearcă și 64, 128)")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--hits",     default=None,
                    help="Fișier .pkl cu hits pre-acumulate (opțional)")
    ap.add_argument("--out",      default="l8_v3_ppo_curve.npz")
    args = ap.parse_args()

    print("=" * 64)
    print(f"L8 V3 PPO — obs 35 dims, {args.steps} pași/episod")
    print("=" * 64)

    initial_hits = set()
    if args.hits and Path(args.hits).exists():
        with open(args.hits, "rb") as f:
            initial_hits = pickle.load(f)
        print(f"\n  Hits pre-încărcate: {len(initial_hits):,} (din {args.hits})")

    env = IbexL8V3Env(
        episode_steps=args.steps,
        seed=args.seed,
        initial_hits=initial_hits,
    )

    s = run_raw([0x00000013] * 8)
    total_tog    = s.by_kind["toggle"][1] if s else 1
    baseline_pct = 100. * len(initial_hits) / max(total_tog, 1)
    env._total_tog = total_tog

    print(f"  Episoade:      {args.episodes}")
    print(f"  Pași/episod:   {args.steps}")
    print(f"  Obs dims:      35  (vs 10 în v2)")
    print(f"  Baseline:      {baseline_pct:.2f}%")
    print(f"  Ceiling L8:   ~74.72%")
    print(f"\n{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δbaseline':>10} | worst module")
    print("-" * 72)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=args.steps,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.08,
        policy_kwargs=dict(net_arch=[512, 256]),
        verbose=0, seed=args.seed, device="cpu",
    )

    cb = Log(baseline_pct)
    t0 = time.time()
    model.learn(total_timesteps=args.episodes * args.steps, callback=cb)
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed/60:.1f} min")

    if cb.history:
        best  = max(h["cum_pct"] for h in cb.history)
        final = cb.history[-1]["cum_pct"]
        print(f"\nRezultate finale:")
        print(f"  Baseline:          {baseline_pct:.2f}%")
        print(f"  V3 PPO best:       {best:.2f}%")
        print(f"  V3 PPO final:      {final:.2f}%")
        print(f"  Câștig:           {final - baseline_pct:+.2f} pp")
        print(f"  Ceiling L8:       ~74.72%")

        eps    = np.array([h["ep"]      for h in cb.history])
        cum    = np.array([h["cum_pct"] for h in cb.history])
        ep_pct = np.array([h["ep_pct"]  for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
