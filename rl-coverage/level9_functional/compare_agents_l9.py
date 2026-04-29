"""compare_agents_l9.py — Compară bins acoperite de PPO vs Constrained Random.

Rulează ambii agenți pe shadow (rapid) și arată:
  - bins acoperite doar de constrained random
  - bins acoperite doar de PPO
  - bins acoperite de amândoi
  - bins acoperite dacă combini ambii

Usage:
    python compare_agents_l9.py --episodes 50
"""

import argparse, sys, random
from pathlib import Path
from collections import defaultdict
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from env_l9_shadow import IbexL9ShadowEnv, _TOTAL_BINS
from constrained_random_l9 import ConstrainedRandomAgent
from codec_l8 import N_OPS, IMM_BUCKETS

try:
    from stable_baselines3 import PPO
    HAS_PPO = True
except ImportError:
    HAS_PPO = False


def run_agent_shadow(agent_type, n_episodes, steps, model=None):
    """Rulează agentul pe shadow și returnează setul de bins acoperite."""
    env = IbexL9ShadowEnv(episode_steps=steps)
    all_hits = set()

    if agent_type == "constrained":
        agent = ConstrainedRandomAgent(episode_steps=steps)
        rng = np.random.default_rng(42)
        for ep in range(n_episodes):
            obs, _ = env.reset()
            agent.reset()
            done = False
            while not done:
                action = agent.sample()
                obs, _, _, done, info = env.step(action)
            ep_hits = set(info.get("ep_hits", []))
            all_hits |= ep_hits
        return all_hits

    elif agent_type == "random":
        rng = np.random.default_rng(99)
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = [rng.integers(N_OPS), rng.integers(32),
                          rng.integers(32), rng.integers(32),
                          rng.integers(IMM_BUCKETS)]
                obs, _, _, done, info = env.step(action)
            ep_hits = set(info.get("ep_hits", []))
            all_hits |= ep_hits
        return all_hits

    elif agent_type == "ppo" and model is not None:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, _, done, info = env.step(action)
            ep_hits = set(info.get("ep_hits", []))
            all_hits |= ep_hits
        return all_hits

    return set()


def categorize(bins):
    cats = defaultdict(set)
    for b in bins:
        if b.startswith("seen_"):    cats["SEEN"].add(b)
        elif b.startswith("opair_"): cats["OPERAND_PAIR"].add(b)
        elif b.startswith("raw"):    cats["RAW_HAZARD"].add(b)
        elif b.startswith("seq_"):   cats["SEQUENCE"].add(b)
        elif b.startswith("corner_"):cats["CORNER"].add(b)
        else:                        cats["OTHER"].add(b)
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--model",    default="l9_shadow_final")
    args = ap.parse_args()

    # Check IbexL9ShadowEnv has ep_hits in info
    # (needs to expose per-episode bin hits)
    env_test = IbexL9ShadowEnv(episode_steps=4)
    obs, _ = env_test.reset()
    rng = np.random.default_rng(0)
    done = False
    while not done:
        a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
             rng.integers(32), rng.integers(IMM_BUCKETS)]
        obs, _, _, done, info = env_test.step(a)
    if "ep_hits" not in info:
        print("[WARN] IbexL9ShadowEnv does not expose 'ep_hits' in info.")
        print("       Run on shadow cumulative hits instead.")
        use_cum = True
    else:
        use_cum = False

    print(f"Running {args.episodes} episodes × {args.steps} steps on shadow...")
    print()

    # ── Constrained Random ────────────────────────────────────────────────
    print("1/3  Constrained Random...")
    env_cr = IbexL9ShadowEnv(episode_steps=args.steps)
    agent_cr = ConstrainedRandomAgent(episode_steps=args.steps)
    for ep in range(args.episodes):
        obs, _ = env_cr.reset()
        agent_cr.reset()
        done = False
        while not done:
            action = agent_cr.sample()
            obs, _, _, done, info = env_cr.step(action)
    hits_cr = set(env_cr._cum_hits)
    total_bins = _TOTAL_BINS
    print(f"   Constrained Random: {len(hits_cr)} bins  "
          f"({100*len(hits_cr)/total_bins:.1f}%)")

    # ── Pure Random ───────────────────────────────────────────────────────
    print("2/3  Pure Random...")
    env_rnd = IbexL9ShadowEnv(episode_steps=args.steps)
    rng = np.random.default_rng(42)
    for ep in range(args.episodes):
        obs, _ = env_rnd.reset()
        done = False
        while not done:
            action = [int(rng.integers(N_OPS)), int(rng.integers(32)),
                      int(rng.integers(32)), int(rng.integers(32)),
                      int(rng.integers(IMM_BUCKETS))]
            obs, _, _, done, info = env_rnd.step(action)
    hits_rnd = set(env_rnd._cum_hits)
    print(f"   Pure Random:        {len(hits_rnd)} bins  "
          f"({100*len(hits_rnd)/total_bins:.1f}%)")

    # ── PPO ───────────────────────────────────────────────────────────────
    hits_ppo = set()
    model_path = THIS / args.model
    if HAS_PPO and (model_path.with_suffix(".zip").exists() or model_path.exists()):
        print("3/3  PPO...")
        model = PPO.load(str(model_path))
        env_ppo = IbexL9ShadowEnv(episode_steps=args.steps)
        for ep in range(args.episodes):
            obs, _ = env_ppo.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, _, done, info = env_ppo.step(action)
        hits_ppo = set(env_ppo._cum_hits)
        print(f"   PPO:               {len(hits_ppo)} bins  "
              f"({100*len(hits_ppo)/total_bins:.1f}%)")
    else:
        print("3/3  PPO model not found, skipping.")

    # ── Comparație ────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("COMPARISON  (shadow, cumulative)")
    print("=" * 55)

    only_cr  = hits_cr  - hits_ppo - hits_rnd
    only_ppo = hits_ppo - hits_cr  - hits_rnd
    only_rnd = hits_rnd - hits_cr  - hits_ppo
    cr_and_ppo = hits_cr & hits_ppo
    all_three  = hits_cr & hits_ppo & hits_rnd
    combined   = hits_cr | hits_ppo | hits_rnd
    combined_cr_ppo = hits_cr | hits_ppo

    print(f"  Constrained Random only : {len(only_cr):>5}  "
          f"({100*len(only_cr)/total_bins:.1f}%)")
    if hits_ppo:
        print(f"  PPO only               : {len(only_ppo):>5}  "
              f"({100*len(only_ppo)/total_bins:.1f}%)")
        print(f"  Random only            : {len(only_rnd):>5}  "
              f"({100*len(only_rnd)/total_bins:.1f}%)")
        print(f"  CR ∩ PPO               : {len(cr_and_ppo):>5}  "
              f"({100*len(cr_and_ppo)/total_bins:.1f}%)")
        print(f"  CR ∪ PPO (combined)    : {len(combined_cr_ppo):>5}  "
              f"({100*len(combined_cr_ppo)/total_bins:.1f}%)")
    print(f"  All three combined     : {len(combined):>5}  "
          f"({100*len(combined)/total_bins:.1f}%)")

    # ── Breakdown pe categorii ────────────────────────────────────────────
    print()
    print("BREAKDOWN BY CATEGORY")
    print("-" * 55)
    cats_cr  = categorize(hits_cr)
    cats_ppo = categorize(hits_ppo)
    cats_rnd = categorize(hits_rnd)

    all_cats = ["SEEN", "OPERAND_PAIR", "RAW_HAZARD", "SEQUENCE", "CORNER", "OTHER"]
    print(f"  {'Category':<15} {'CR':>6} {'PPO':>6} {'Rnd':>6} {'CR∪PPO':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for cat in all_cats:
        cr_n   = len(cats_cr.get(cat, set()))
        ppo_n  = len(cats_ppo.get(cat, set())) if hits_ppo else 0
        rnd_n  = len(cats_rnd.get(cat, set()))
        comb_n = len(cats_cr.get(cat, set()) | cats_ppo.get(cat, set()))
        print(f"  {cat:<15} {cr_n:>6} {ppo_n:>6} {rnd_n:>6} {comb_n:>8}")

    # ── Bins acoperite doar de PPO (interesante!) ─────────────────────────
    if hits_ppo and only_ppo:
        print()
        print(f"BINS COVERED ONLY BY PPO  ({len(only_ppo)} total) — first 20:")
        for b in sorted(only_ppo)[:20]:
            print(f"   {b}")

    if only_cr:
        print()
        print(f"BINS COVERED ONLY BY CR  ({len(only_cr)} total) — first 20:")
        for b in sorted(only_cr)[:20]:
            print(f"   {b}")

    np.savez(str(THIS / "agent_comparison.npz"),
             hits_cr  = np.array(sorted(hits_cr)),
             hits_ppo = np.array(sorted(hits_ppo)),
             hits_rnd = np.array(sorted(hits_rnd)))
    print()
    print(f"Saved agent_comparison.npz")


if __name__ == "__main__":
    main()
