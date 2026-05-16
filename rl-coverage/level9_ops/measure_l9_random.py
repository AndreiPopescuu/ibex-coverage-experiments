"""Random-agent baseline pe L9 (83 ops).

Comparabil direct cu L8 random (66.07% la 30 episoade × 256 steps).
"""

import time, sys
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l9_v2 import IbexL9V2Env
from codec_l9 import N_OPS, IMM_BUCKETS

EP_STEPS   = 256
N_EPISODES = 30
OUT_NPZ    = Path(__file__).parent / "l9_random_baseline.npz"


def main():
    env = IbexL9V2Env(episode_steps=EP_STEPS, seed=42)
    rng = np.random.default_rng(42)

    print(f"L9 random baseline — {N_OPS} ops × 32³ × {IMM_BUCKETS} imm, "
          f"{EP_STEPS} steps/ep, {N_EPISODES} ep")
    print(f"{'ep':>3} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'wall':>5}")
    print("-" * 42)

    ep_pcts  = np.zeros(N_EPISODES)
    cum_pcts = np.zeros(N_EPISODES)

    for ep in range(N_EPISODES):
        env.reset()
        t0 = time.time()
        for _ in range(EP_STEPS):
            a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
                 rng.integers(32), rng.integers(IMM_BUCKETS)]
            _, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        dt = time.time() - t0
        ep_pcts[ep]  = info.get("ep_pct", 0.0)
        cum_pcts[ep] = info.get("cum_pct", 0.0)
        new_hits     = info.get("new_hits_vs_cum", 0)
        print(f"{ep+1:>3} | {ep_pcts[ep]:>5.2f}% | {cum_pcts[ep]:>5.2f}% | "
              f"{new_hits:>5} | {dt:>4.1f}s", flush=True)

    np.savez(OUT_NPZ, ep=np.arange(1, N_EPISODES + 1),
             ep_pct=ep_pcts, cum_pct=cum_pcts)

    print(f"\nSaved → {OUT_NPZ.name}")
    print(f"L9 random final cum: {cum_pcts[-1]:.2f}%")
    print(f"L8 random baseline:   66.07%  (referință)")


if __name__ == "__main__":
    main()
