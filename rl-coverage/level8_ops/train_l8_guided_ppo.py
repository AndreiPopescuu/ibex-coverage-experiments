"""train_l8_guided_ppo.py — PPO cu observation îmbogățit + reward shaping.

Pornește de la hits-urile acumulate de pipeline_l8.py (random + CDG),
și antrenează PPO care:
  - Vede per-modul coverage în observație (știe ce module sunt slab acoperite)
  - Primește reward mai mare pentru hits în module slab acoperite
  - Încearcă să depășească plafonul CDG

Usage:
    python train_l8_guided_ppo.py --hits l8_pipeline_hits.pkl --episodes 30
"""

import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l8_guided import IbexL8GuidedEnv, MODULES

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
        ep  = len(self.history) + 1
        cum = info["cum_pct"]
        new = info.get("new_hits_vs_cum", 0)
        br  = info.get("branch_pct", 0.0)
        mod = info.get("mod_coverage", {})
        self.history.append({"ep": ep, "cum_pct": cum, "ep_pct": info["ep_pct"],
                              "new_hits": new, "branch": br})
        delta = cum - self.baseline_pct

        # Modul cu coverage minim
        worst_mod = min(mod, key=mod.get) if mod else "?"
        worst_pct = mod.get(worst_mod, 0) * 100

        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | cum {cum:>5.2f}% | "
              f"new {new:>4} | Δ {delta:>+6.2f}pp | worst: {worst_mod} {worst_pct:.1f}%",
              flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits",     default=None,
                    help="Fișier .pkl cu hits din CDG pipeline (omite pentru start de la 0)")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--steps",    type=int, default=1024)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="l8_guided_ppo_curve.npz")
    args = ap.parse_args()

    # Încarcă hits din CDG
    print("=" * 64)
    print("L8 Guided PPO — per-modul obs + reward shaping")
    print("=" * 64)

    initial_hits = set()
    if args.hits and Path(args.hits).exists():
        with open(args.hits, "rb") as f:
            initial_hits = pickle.load(f)
        print(f"\n  Hits încărcate: {len(initial_hits):,} (din {args.hits})")
    else:
        print(f"\n  [WARN] {args.hits} nu există — pornire de la 0")

    env = IbexL8GuidedEnv(
        episode_steps=args.steps,
        seed=args.seed,
        initial_hits=initial_hits,
    )

    # Obținem baseline pct dintr-un run minim dacă nu știm total_tog
    from directed_l8 import run_raw
    s = run_raw([0x00000013] * 8)
    if s:
        total_tog = s.by_kind["toggle"][1]
        baseline_pct = 100. * len(initial_hits) / max(total_tog, 1)
        env._total_tog = total_tog
    else:
        baseline_pct = 0.

    print(f"  Baseline CDG:   {baseline_pct:.2f}%")
    print(f"  Ceiling L8:    ~74.72%")
    print(f"  Gap:            {74.72 - baseline_pct:.2f} pp")
    print(f"\n  Module urmărite + weights:")
    for m in MODULES:
        from env_l8_guided import MODULE_WEIGHTS
        print(f"    {m:<25s} weight={MODULE_WEIGHTS[m]:.1f}x")

    print(f"\nPPO guided — {args.episodes} ep × {args.steps} pași")
    print(f"{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δbaseline':>10} | worst module")
    print("-" * 70)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=args.steps,
        batch_size=256,
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
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
        print(f"  Baseline (random+CDG):     {baseline_pct:.2f}%")
        print(f"  Guided PPO best:           {best:.2f}%")
        print(f"  Guided PPO final:          {final:.2f}%")
        print(f"  Câștig față de CDG:       {final - baseline_pct:+.2f} pp")
        print(f"  Ceiling L8:               ~74.72%")

        eps    = np.array([h["ep"]       for h in cb.history])
        cum    = np.array([h["cum_pct"]  for h in cb.history])
        ep_pct = np.array([h["ep_pct"]   for h in cb.history])
        branch = np.array([h["branch"]   for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct, branch_pct=branch)
        print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
