"""collect_l8_hits.py — Acumulează hits L8 până la ~70% și salvează în .pkl.

Rulează episoade cu politică random până coverage-ul cumulativ se plafonează
(același prag atins de PPO L8 v2: ~70.7%).

Usage:
    python collect_l8_hits.py
    python collect_l8_hits.py --out l8_final_hits.pkl --target 70.5
"""

import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l8_dynamic import IbexL8DynamicEnv
from codec_l8 import N_OPS, IMM_BUCKETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",      default="l8_final_hits.pkl")
    ap.add_argument("--target",   type=float, default=70.5,
                    help="Oprește când cum% atinge această valoare")
    ap.add_argument("--max-eps",  type=int, default=2000)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = IbexL8DynamicEnv(episode_steps=args.steps)

    print(f"Colectez hits L8 până la {args.target}% (max {args.max_eps} ep)...")
    print(f"{'ep':>5} | {'cum%':>6} | {'new':>5} | {'wall':>5}")
    print("-" * 32)

    for ep in range(1, args.max_eps + 1):
        env.reset()
        t0 = time.time()
        for _ in range(args.steps):
            action = [rng.integers(N_OPS), rng.integers(32),
                      rng.integers(32), rng.integers(32),
                      rng.integers(IMM_BUCKETS)]
            _, _, _, trunc, info = env.step(action)
            if trunc:
                break
        dt = time.time() - t0
        cum = info.get("cum_pct", 0.0)
        new = info.get("new_hits_vs_cum", 0)
        print(f"  ep {ep:>4} | {cum:>5.2f}% | {new:>5} | {dt:.1f}s", flush=True)

        if cum >= args.target:
            break

    hits = env._cum_hits
    out  = THIS / args.out
    with open(out, "wb") as f:
        pickle.dump(hits, f)

    print(f"\nSalvat {len(hits):,} hits → {out}")
    print(f"Coverage final: {cum:.2f}%")


if __name__ == "__main__":
    main()
