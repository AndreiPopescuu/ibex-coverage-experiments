"""pipeline_l8.py — Pipeline adaptiv în 3 faze pentru L8.

Faza 1: Constrained Random
  - Rulează N episoade × 1024 instrucțiuni random
  - Acumulează baseline coverage

Faza 2: Greedy CDG (cu early stopping)
  - Rulează runde de CDG (5 candidați × 512 pași)
  - Se oprește automat dacă nu găsește nimic nou în 7 runde consecutive
  - Predă hits acumulate fazei 3

Faza 3: PPO Novelty
  - Pornește de la tot ce au acumulat fazele 1+2
  - Reward = DOAR hits noi față de baseline acumulat
  - Rulează N episoade și vedem dacă mai îmbunătățește coverage-ul

Usage:
    python pipeline_l8.py
    python pipeline_l8.py --random-eps 5 --cdg-rounds 50 --ppo-eps 30
"""

import argparse, sys, time, pickle
from pathlib import Path
import numpy as np

THIS  = Path(__file__).resolve().parent
L5    = (THIS.parent / "level5_real_rtl").resolve()
ML4DV = (THIS.parent.parent / "cpu").resolve()
COVDAT = ML4DV / "coverage.dat"

sys.path.insert(0, str(L5))
import cov_parser

from env_l8 import IbexL8Env
from directed_l8 import run_raw, cum_hits_from
from greedy_cdg_l8 import random_program, mutate, eval_candidate
from codec_l8 import N_OPS, IMM_BUCKETS

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


# ── PPO Callback ──────────────────────────────────────────────────────────

class PPOLog(BaseCallback):
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
            "ep": ep, "ep_pct": info["ep_pct"],
            "cum_pct": cum_pct, "new_hits": info.get("new_hits_vs_cum", 0),
        })
        delta = cum_pct - self.baseline_pct
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | "
              f"cum {cum_pct:>5.2f}% | new {info.get('new_hits_vs_cum',0):>4} | "
              f"Δbaseline {delta:+.2f}pp", flush=True)
        return True


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-eps",   type=int, default=5,
                    help="Episoade constrained random (Faza 1)")
    ap.add_argument("--cdg-rounds",   type=int, default=50,
                    help="Runde CDG maxime (Faza 2, cu early stopping la 7 runde fără hits)")
    ap.add_argument("--cdg-patience", type=int, default=7,
                    help="Runde consecutive fără hits noi → oprire CDG")
    ap.add_argument("--cdg-candidates", type=int, default=5)
    ap.add_argument("--cdg-steps",    type=int, default=512)
    ap.add_argument("--ppo-eps",      type=int, default=30,
                    help="Episoade PPO novelty (Faza 3)")
    ap.add_argument("--ppo-steps",    type=int, default=1024)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--out",          default="l8_pipeline_curve.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 64)
    print("L8 Pipeline: Constrained Random → CDG → PPO Novelty")
    print(f"  Faza 1: {args.random_eps} ep random")
    print(f"  Faza 2: max {args.cdg_rounds} runde CDG (stop la {args.cdg_patience} runde fără hits)")
    print(f"  Faza 3: {args.ppo_eps} ep PPO novelty")
    print("=" * 64)

    cum_hits  = set()
    total_tog = None
    history   = {"random": [], "cdg": [], "ppo": []}

    # ═══════════════════════════════════════════════════════════════════════
    # FAZA 1: Constrained Random
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print(f"FAZA 1 — Constrained Random ({args.random_eps} ep × 1024 pași)")
    print(f"{'='*64}")

    env = IbexL8Env(episode_steps=1024, seed=args.seed, reward_mode="novelty")
    rng_r = np.random.default_rng(args.seed + 1)

    for ep in range(args.random_eps):
        env.reset()
        t0 = time.time()
        for _ in range(1024):
            a = [rng_r.integers(N_OPS), rng_r.integers(32),
                 rng_r.integers(32), rng_r.integers(32),
                 rng_r.integers(IMM_BUCKETS)]
            _, _, _, trunc, _ = env.step(a)
            if trunc: break
        s = cov_parser.parse(str(COVDAT))
        hits = cum_hits_from(s)
        new_h = hits - cum_hits
        cum_hits |= hits
        if total_tog is None:
            total_tog = s.by_kind["toggle"][1]
        cum_pct = 100. * len(cum_hits) / total_tog
        dt = time.time() - t0
        history["random"].append(cum_pct)
        print(f"  ep {ep+1:>3} | new={len(new_h):>5} | cum={cum_pct:.2f}%  ({dt:.0f}s)", flush=True)

    random_baseline = 100. * len(cum_hits) / total_tog
    print(f"\n  Faza 1 final: {random_baseline:.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # FAZA 2: Greedy CDG cu early stopping
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print(f"FAZA 2 — Greedy CDG (max {args.cdg_rounds} runde, patience={args.cdg_patience})")
    print(f"{'='*64}")
    print(f"{'rnd':>4} | {'best_new':>8} | {'tot_new':>8} | {'cum%':>7} | {'Δrandom':>8} | {'s':>5}")
    print("-" * 55)

    seed_program = None
    zero_streak  = 0
    cdg_baseline = random_baseline

    for rnd in range(1, args.cdg_rounds + 1):
        t0 = time.time()
        candidates = []
        for _ in range(args.cdg_candidates):
            if seed_program is None or rng.random() < 0.5:
                prog = random_program(rng, args.cdg_steps)
            else:
                prog = mutate(seed_program, rng)
            candidates.append(prog)

        best_new  = 0
        best_prog = None
        round_hits = set()

        for prog in candidates:
            n_new, hits = eval_candidate(prog, cum_hits)
            round_hits |= hits
            if n_new > best_new:
                best_new  = n_new
                best_prog = prog

        new_total = len(round_hits - cum_hits)
        cum_hits |= round_hits

        if best_prog is not None:
            seed_program = best_prog

        cum_pct = 100. * len(cum_hits) / total_tog
        dt = time.time() - t0
        history["cdg"].append(cum_pct)

        print(f"{rnd:>4} | {best_new:>8} | {new_total:>8} | {cum_pct:>6.2f}% | "
              f"{cum_pct - cdg_baseline:>+7.2f}pp | {dt:>4.0f}s", flush=True)

        # Early stopping
        if new_total == 0:
            zero_streak += 1
            if zero_streak >= args.cdg_patience:
                print(f"\n  [Early stop] {args.cdg_patience} runde fără hits noi. Oprire CDG.")
                break
        else:
            zero_streak = 0

    cdg_final = 100. * len(cum_hits) / total_tog
    print(f"\n  Faza 2 final: {cdg_final:.2f}% (câștig CDG: {cdg_final - random_baseline:+.2f}pp)")

    # Salvează hits pentru referință
    with open("l8_pipeline_hits.pkl", "wb") as f:
        pickle.dump(cum_hits, f)

    # ═══════════════════════════════════════════════════════════════════════
    # FAZA 3: PPO Novelty
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print(f"FAZA 3 — PPO Novelty ({args.ppo_eps} ep × {args.ppo_steps} pași)")
    print(f"  Baseline (random+CDG): {cdg_final:.2f}%")
    print(f"  Ceiling L8:           ~74.72%")
    print(f"  Gap rămas:             {74.72 - cdg_final:.2f} pp")
    print(f"{'='*64}")
    print(f"{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δbaseline':>10}")
    print("-" * 45)

    ppo_env = IbexL8Env(
        episode_steps=args.ppo_steps,
        seed=args.seed,
        reward_mode="novelty",
    )
    ppo_env._cum_hits = set(cum_hits)

    model = PPO(
        "MlpPolicy", ppo_env,
        learning_rate=3e-4,
        n_steps=args.ppo_steps,
        batch_size=256,
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0, seed=args.seed, device="cpu",
    )

    cb = PPOLog(cdg_final)
    t0 = time.time()
    model.learn(total_timesteps=args.ppo_eps * args.ppo_steps, callback=cb)
    elapsed = time.time() - t0

    if cb.history:
        for h in cb.history:
            history["ppo"].append(h["cum_pct"])
        ppo_final = cb.history[-1]["cum_pct"]
        ppo_best  = max(h["cum_pct"] for h in cb.history)
    else:
        ppo_final = cdg_final
        ppo_best  = cdg_final

    print(f"\nFaza 3 done în {elapsed/60:.1f} min")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMAR FINAL
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print(f"SUMAR PIPELINE L8")
    print(f"{'='*64}")
    print(f"  Faza 1 — Constrained Random:  {random_baseline:.2f}%")
    print(f"  Faza 2 — + CDG:               {cdg_final:.2f}%  ({cdg_final-random_baseline:+.2f}pp)")
    print(f"  Faza 3 — + PPO novelty:        {ppo_final:.2f}%  ({ppo_final-cdg_final:+.2f}pp)")
    print(f"  Ceiling L8:                   ~74.72%")
    print(f"  Distanță față de ceiling:      {74.72-ppo_final:.2f}pp")

    np.savez(args.out,
             random_curve = np.array(history["random"]),
             cdg_curve    = np.array(history["cdg"]),
             ppo_curve    = np.array(history["ppo"]),
             random_final = np.array([random_baseline]),
             cdg_final    = np.array([cdg_final]),
             ppo_final    = np.array([ppo_final]))
    print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
