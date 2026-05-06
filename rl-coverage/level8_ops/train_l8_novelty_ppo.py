"""train_l8_novelty_ppo.py — PPO cu novelty reward, pornind de la baseline directed.

Strategia:
  1. Rulează programele diriguite (misaligned, counter, multdiv, csr) o dată
     și pre-acumulează hits-urile lor în env._cum_hits
  2. Antrenează PPO cu reward = ONLY new hits (novelty) față de baseline
  3. Raportează câte pp în plus față de directed+random baseline

Baseline de bătut: 66.62% (random 5 ep + directed programs)

Usage:
    python train_l8_novelty_ppo.py --episodes 30   # ~25 min
    python train_l8_novelty_ppo.py --episodes 60   # ~50 min
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
L5   = (THIS.parent / "level5_real_rtl").resolve()
ML4DV = (THIS.parent.parent / "cpu").resolve()
COVDAT = ML4DV / "coverage.dat"

sys.path.insert(0, str(L5))
import cov_parser

from env_l8 import IbexL8Env, run_program
from directed_l8 import (
    run_raw, cum_hits_from,
    prog_misaligned, prog_counter_writes,
    prog_multdiv_diverse, prog_csr_bitwalking, nop,
)
from codec_l8 import N_OPS, IMM_BUCKETS

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


# ── Callback ──────────────────────────────────────────────────────────────

class Log(BaseCallback):
    def __init__(self, baseline_pct: float):
        super().__init__()
        self.baseline_pct = baseline_pct
        self.history = []

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "ep_pct" not in info:
            return True
        ep_total = info.get("ep_total", 1)
        cum_pct  = 100.0 * info.get("cum_covered", 0) / ep_total if ep_total else 0.0
        ep = len(self.history) + 1
        self.history.append({
            "ep":       ep,
            "ep_pct":   info["ep_pct"],
            "cum_pct":  cum_pct,
            "new_hits": info.get("new_hits_vs_cum", 0),
            "branch":   info.get("branch_pct", 0.0),
        })
        delta = cum_pct - self.baseline_pct
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | "
              f"cum {cum_pct:>5.2f}% | new {info.get('new_hits_vs_cum',0):>4} | "
              f"Δbaseline {delta:+.2f}pp", flush=True)
        return True


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",      type=int, default=30)
    ap.add_argument("--episode-steps", type=int, default=1024)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--out",           default="l8_novelty_ppo_curve.npz")
    ap.add_argument("--skip-directed", action="store_true",
                    help="Sare peste warm-up directed (mai rapid, baseline mai mic)")
    ap.add_argument("--load-hits", default=None,
                    help="Fișier .pkl cu hits pre-acumulate (de la CDG). Sare peste warm-up.")
    args = ap.parse_args()

    # ── Pas 1: pre-acumulare baseline ────────────────────────────────────
    print("=" * 64)
    print("L8 Novelty PPO — pornind de la baseline directed+random")
    print("=" * 64)

    cum_hits   = set()
    total_tog  = None

    if args.load_hits:
        import pickle
        with open(args.load_hits, "rb") as f:
            cum_hits = pickle.load(f)
        # Obținem total_tog dintr-un run minim
        s = run_raw([0x00000013] * 8)
        if s:
            total_tog = s.by_kind["toggle"][1]
        baseline_pct = 100. * len(cum_hits) / total_tog if total_tog else 0.
        print(f"\n  Hits încărcate din {args.load_hits}")
        print(f"  Baseline pre-acumulat: {baseline_pct:.2f}%")
    elif not args.skip_directed:
        # Random warm-up: 5 episoade
        print("\n[1/2] Random warm-up (5 ep × 1024 pași)...")
        env_warmup = IbexL8Env(episode_steps=1024, seed=99, reward_mode="novelty")
        rng = np.random.default_rng(99)
        for ep in range(5):
            env_warmup.reset()
            for _ in range(1024):
                a = [rng.integers(N_OPS), rng.integers(32), rng.integers(32),
                     rng.integers(32), rng.integers(IMM_BUCKETS)]
                _, _, _, trunc, _ = env_warmup.step(a)
                if trunc: break
            s = cov_parser.parse(str(COVDAT))
            cum_hits |= cum_hits_from(s)
            if total_tog is None:
                total_tog = s.by_kind["toggle"][1]
            print(f"  ep {ep+1}: cum={100.*len(cum_hits)/total_tog:.2f}%", flush=True)

        # Directed programs
        print("\n[2/2] Directed warm-up (4 programe)...")
        programs = [
            ("Misaligned",     prog_misaligned()),
            ("Counter writes", prog_counter_writes()),
            ("MUL/DIV",        prog_multdiv_diverse()),
            ("CSR walking",    prog_csr_bitwalking()),
        ]
        for name, prog in programs:
            s = run_raw(prog)
            if s is None:
                print(f"  [FAIL] {name}"); continue
            hits    = cum_hits_from(s)
            new_h   = hits - cum_hits
            cum_hits |= hits
            cum_pct = 100. * len(cum_hits) / total_tog
            print(f"  {name:<20s}: +{len(new_h):>5} hits → cum={cum_pct:.2f}%", flush=True)
    elif args.skip_directed:
        # Fără warm-up — rulăm un episod minim să obținem total_tog
        print("\n[skip] Fără directed warm-up.")
        s = run_raw([0x00000013] * 10)   # 10 NOPs
        if s:
            total_tog = s.by_kind["toggle"][1]

    baseline_pct = 100. * len(cum_hits) / total_tog if total_tog else 0.
    print(f"\n  Baseline acumulat: {baseline_pct:.2f}%")
    print(f"  Ceiling L8:       ~74.72%")
    print(f"  Gap de închis:     {74.72 - baseline_pct:.2f} pp\n")

    # ── Pas 2: PPO cu novelty reward pornind de la baseline ──────────────
    print(f"PPO novelty — {args.episodes} ep × {args.episode_steps} pași")
    print(f"{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δbaseline':>10}")
    print("-" * 50)

    env = IbexL8Env(
        episode_steps=args.episode_steps,
        seed=args.seed,
        reward_mode="novelty",
    )
    # Pre-injectăm hits acumulate din warm-up
    env._cum_hits = set(cum_hits)

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

    cb  = Log(baseline_pct)
    t0  = time.time()
    model.learn(
        total_timesteps=args.episodes * args.episode_steps,
        callback=cb,
    )
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed/60:.1f} min")

    if cb.history:
        eps    = np.array([h["ep"]       for h in cb.history])
        cum    = np.array([h["cum_pct"]  for h in cb.history])
        ep_pct = np.array([h["ep_pct"]   for h in cb.history])
        branch = np.array([h["branch"]   for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct, branch_pct=branch)

        best = cum.max()
        final = cum[-1]
        print(f"\nRezultate finale:")
        print(f"  Baseline (directed+random): {baseline_pct:.2f}%")
        print(f"  PPO best cum:               {best:.2f}%")
        print(f"  PPO final cum:              {final:.2f}%")
        print(f"  Câștig PPO vs baseline:    {final - baseline_pct:+.2f} pp")
        print(f"  Ceiling L8:               ~74.72%")
        print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
