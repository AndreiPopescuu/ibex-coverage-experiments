"""Train PPO on the L8 rich-obs env (70-op codec).

Direct comparison with the L8 random baseline (l8_random_baseline.npz):
same stimulus space, same episode length, only difference is PPO vs random.

Usage:
    python train_l8_ppo.py                   # 150 episodes (default)
    python train_l8_ppo.py --episodes 300    # longer run
    python train_l8_ppo.py --episodes 30     # quick comparison with random baseline
"""

import argparse, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env_l8_rich import IbexL8RichEnv


class Log(BaseCallback):
    def __init__(self): super().__init__(); self.history = []

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "cum_pct" not in info:
            return True
        ep = len(self.history) + 1
        self.history.append({
            "ep": ep,
            "ep_pct": info["ep_pct"],
            "cum_pct": info["cum_pct"],
            "new_hits": info.get("new_hits", 0),
            "branch_pct": info.get("branch_pct", 0.0),
        })
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | "
              f"cum {info['cum_pct']:>5.2f}% | new {info['new_hits']:>4} | "
              f"branch {info['branch_pct']:>5.2f}%", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--episode-steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="l8_ppo_curve.npz")
    args = ap.parse_args()

    env = IbexL8RichEnv(episode_steps=args.episode_steps, seed=args.seed,
                        reward_mode="novelty")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=args.episode_steps,
        batch_size=256,
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0, seed=args.seed, device="cpu",
    )
    print(f"Training PPO on L8 env for {args.episodes} episodes...")
    print(f"L8 random baseline (30 eps): see l8_random_baseline.npz")
    print(f"L7 random baseline (30 eps): 66.27%\n")
    cb = Log()
    t0 = time.time()
    model.learn(total_timesteps=args.episodes * args.episode_steps, callback=cb)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    if cb.history:
        eps = np.array([h["ep"] for h in cb.history])
        cum = np.array([h["cum_pct"] for h in cb.history])
        ep_pct = np.array([h["ep_pct"] for h in cb.history])
        branch = np.array([h["branch_pct"] for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct, branch_pct=branch)
        print(f"Saved {args.out}")
        print(f"Final cum toggle: {cum[-1]:.2f}%")
        print(f"L7 random (30 eps):  66.27%")
        print(f"L8 random (30 eps):  see l8_random_baseline.npz")


if __name__ == "__main__":
    main()
