"""
Train PPO on the UNSTRUCTURED decoder env and compare with structured PPO + random.

The agent gets 4 raw bytes as action (no field knowledge) — same level of
"help" as ML4DV's LLM which also generates raw hex instructions.
"""

import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from decoder_env import IbexDecoderEnv
from decoder_env_raw import IbexDecoderRawEnv, decode_raw
from shadow_decoder import N_BINS

SEED         = 42
PPO_STEPS    = 300_000
EVAL_SAMPLES = 10_000
EP_STEPS     = 512


def random_raw(n, seed=0):
    """Random baseline: uniform 32-bit integers (same as ML4DV raw baseline)."""
    from shadow_decoder import bins_for_action
    from decoder_env_raw import decode_raw
    rng = np.random.default_rng(seed)
    covered = np.zeros(N_BINS, dtype=np.int8)
    curve = np.empty(n, dtype=np.int32)
    for t in range(n):
        word = int(rng.integers(0, 1 << 32))
        decoded = decode_raw(word)
        if decoded is not None:
            op_idx, rd, rs1, rs2 = decoded
            for b in bins_for_action(op_idx, rd, rs1, rs2):
                covered[b] = 1
        curve[t] = int(covered.sum())
    return curve


def eval_env(model, EnvClass, n, seed=0, **kwargs):
    env = EnvClass(episode_steps=n, seed=seed, **kwargs)
    obs, _ = env.reset(seed=seed)
    curve = np.empty(n, dtype=np.int32)
    for t in range(n):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(action)
        curve[t] = info["covered"]
        if term or trunc:
            obs, _ = env.reset()
    return curve


def train(EnvClass, label, **kwargs):
    def make(): return EnvClass(episode_steps=EP_STEPS, seed=SEED, **kwargs)
    vec = DummyVecEnv([make] * 4)
    model = PPO("MlpPolicy", vec,
                learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4,
                gamma=0.99, ent_coef=0.02, verbose=0, seed=SEED, device="cpu")
    print(f"Training {label} ({PPO_STEPS} steps)...")
    t0 = time.time()
    model.learn(total_timesteps=PPO_STEPS, progress_bar=False)
    print(f"  done in {time.time()-t0:.1f}s")
    return model


def main():
    print(f"{'='*60}")
    print(f"Decoder coverage: structured vs raw action space")
    print(f"Eval samples: {EVAL_SAMPLES}, N_BINS: {N_BINS}")
    print(f"{'='*60}\n")

    # 1. Random raw baseline (like ML4DV's random hex approach)
    print("Random raw 32-bit baseline...")
    rand_raw = random_raw(EVAL_SAMPLES, seed=SEED)
    print(f"  {rand_raw[-1]}/{N_BINS} = {100*rand_raw[-1]/N_BINS:.1f}%\n")

    # 2. PPO with UNSTRUCTURED action space (4 raw bytes)
    model_raw = train(IbexDecoderRawEnv, "PPO-RAW (unstructured)")
    ppo_raw = eval_env(model_raw, IbexDecoderRawEnv, EVAL_SAMPLES)
    print(f"  PPO-RAW eval: {ppo_raw[-1]}/{N_BINS} = {100*ppo_raw[-1]/N_BINS:.1f}%\n")

    # 3. PPO with STRUCTURED action space (for reference)
    model_struct = train(IbexDecoderEnv, "PPO-STRUCTURED (op,rd,rs1,rs2)")
    ppo_struct = eval_env(model_struct, IbexDecoderEnv, EVAL_SAMPLES)
    print(f"  PPO-STRUCTURED eval: {ppo_struct[-1]}/{N_BINS} = {100*ppo_struct[-1]/N_BINS:.1f}%\n")

    print(f"\n{'='*60}")
    print(f"SUMMARY ({EVAL_SAMPLES} instructions each)")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Final coverage':>15} {'% of 2107':>10}")
    print(f"{'-'*60}")
    print(f"{'Random raw 32-bit':<30} {rand_raw[-1]:>12}/2107 {100*rand_raw[-1]/N_BINS:>9.1f}%")
    print(f"{'PPO raw (unstructured)':<30} {ppo_raw[-1]:>12}/2107 {100*ppo_raw[-1]/N_BINS:>9.1f}%")
    print(f"{'PPO structured':<30} {ppo_struct[-1]:>12}/2107 {100*ppo_struct[-1]/N_BINS:>9.1f}%")
    print(f"{'ML4DV (Claude 3.5 Sonnet)':<30} {'~2005':>12}/2107 {'95.2':>9}%")
    print(f"{'='*60}")

    np.savez("curves_raw_vs_structured.npz",
             rand_raw=rand_raw, ppo_raw=ppo_raw, ppo_struct=ppo_struct)
    print("Saved curves_raw_vs_structured.npz")


if __name__ == "__main__":
    main()
