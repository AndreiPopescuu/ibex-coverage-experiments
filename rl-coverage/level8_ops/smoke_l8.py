"""Smoke test for L8: codec self-test + 1 random episode."""

import sys, time
from pathlib import Path
import numpy as np

import codec_l8
from env_l8 import IbexL8Env
from codec_l8 import N_OPS, IMM_BUCKETS, Op


def main():
    print("=== codec self-test ===")
    codec_l8._self_test()

    print("\n=== 1-episode random smoke run ===")
    env = IbexL8Env(episode_steps=256, seed=0)
    rng = np.random.default_rng(0)
    env.reset()
    t0 = time.time()
    info = {}
    for _ in range(256):
        a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
             rng.integers(32), rng.integers(IMM_BUCKETS)]
        _, _, _, trunc, info = env.step(a)
        if trunc:
            break
    dt = time.time() - t0
    if info.get("vtop_failed"):
        print("  VTOP FAILED — check sim_build/Vtop exists")
        sys.exit(1)
    print(f"  ep toggle:  {info['ep_pct']:.2f}%  ({info['ep_covered']}/{info['ep_total']})")
    print(f"  branch:     {info['branch_pct']:.2f}%")
    print(f"  line:       {info['line_pct']:.2f}%")
    print(f"  wall time:  {dt:.1f}s")
    print("\n[OK] smoke test passed")


if __name__ == "__main__":
    main()
