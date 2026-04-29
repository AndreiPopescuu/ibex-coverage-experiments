"""greedy_l9.py — Random baseline pentru Rich Functional Coverage.

Usage:
    python greedy_l9.py --episodes 30
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from env_l9 import IbexL9Env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--driver",   default="test_run_for_l9")
    args = ap.parse_args()

    env = IbexL9Env(episode_steps=args.steps, driver=args.driver)

    print(f"Random baseline: {args.episodes} ep × {args.steps} pași")
    print(f"{'ep':>5}  {'ep%':>7}  {'cum%':>8}  {'new':>6}  {'time':>6}")
    print("-"*40)

    ep_pcts = []; cum_pcts = []; new_list = []
    t_total = time.time()

    for ep in range(1, args.episodes+1):
        t0 = time.time()
        obs, _ = env.reset()
        done = False; info = {}
        while not done:
            action = env.action_space.sample()
            obs, _, _, done, info = env.step(action)
        ep_pcts.append(info.get("ep_pct", 0))
        cum_pcts.append(info.get("cum_pct", 0))
        new_list.append(info.get("new_hits", 0))
        print(f"{ep:>5}  {ep_pcts[-1]:>6.1f}%  {cum_pcts[-1]:>7.1f}%  "
              f"{new_list[-1]:>6}  {time.time()-t0:>5.0f}s")

    print("="*40)
    print(f"Final cum coverage: {cum_pcts[-1]:.2f}%")
    print(f"Total bins: {env._total_bins}")
    save = THIS / "l9_random_baseline.npz"
    np.savez(str(save), ep_pct=np.array(ep_pcts), cum_pct=np.array(cum_pcts),
             new_hits=np.array(new_list))
    print(f"[saved] {save}")


if __name__ == "__main__":
    main()
