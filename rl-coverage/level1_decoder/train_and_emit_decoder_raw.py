"""Train PPO on the UNSTRUCTURED decoder env, roll out a program,
save to /tmp/rl_decoder_raw_program.json for real-RTL validation.

The agent generates 4 raw bytes per step — no field structure given.
Same level of prior knowledge as ML4DV's LLM approach.
"""
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from decoder_env_raw import IbexDecoderRawEnv, decode_raw
from shadow_decoder import bins_for_action, BIN_NAMES, N_BINS


def rollout_policy(model, n: int, seed: int):
    env = IbexDecoderRawEnv(episode_steps=n, seed=seed)
    obs, _ = env.reset(seed=seed)
    program, covered = [], set()
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=False)
        b0, b1, b2, b3 = [int(x) for x in action]
        word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        decoded = decode_raw(word)
        if decoded is not None:
            op_idx, rd, rs1, rs2 = decoded
            for b in bins_for_action(op_idx, rd, rs1, rs2):
                covered.add(b)
        program.append(word)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=seed + 1)
    return program, sorted(BIN_NAMES[b] for b in covered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps",    type=int, default=300_000)
    ap.add_argument("--rollout-n",    type=int, default=10_000)
    ap.add_argument("--episode-steps",type=int, default=512)
    ap.add_argument("--out", default="/tmp/rl_decoder_raw_program.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def make_env(): return IbexDecoderRawEnv(episode_steps=args.episode_steps, seed=args.seed)
    vec = DummyVecEnv([make_env for _ in range(4)])
    model = PPO("MlpPolicy", vec,
                learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4,
                gamma=0.99, ent_coef=0.02, verbose=0, seed=args.seed, device="cpu")

    print(f"Training PPO-RAW (unstructured) for {args.ppo_steps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, progress_bar=False)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Rolling out {args.rollout_n} instructions...")
    program, shadow_hit = rollout_policy(model, args.rollout_n, seed=args.seed)
    print(f"  shadow predicts {len(shadow_hit)}/{N_BINS} bins")

    with open(args.out, "w") as f:
        json.dump({"n_instructions": len(program), "seed": args.seed,
                   "mode": "raw", "program": program,
                   "shadow_hit_bins": shadow_hit}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
