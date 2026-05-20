"""train_l10_ppo.py — PPO pe L10 (87 ops), continuare din hits L9.

Configurație identică cu L9 V2:
  - IbexL10Env: 87 ops, obs 32 dims
  - net_arch: [512, 256]
  - ent_coef: 0.08 (default)

Usage:
    # continuare din hits L9 (pipeline normal):
    python train_l10_ppo.py --hits ../level9_ops/l9_v2_checkpoint_hits.pkl --episodes 3600

    # resume din checkpoint L10:
    python train_l10_ppo.py --resume --episodes 3600

    # cu mai multă explorare:
    python train_l10_ppo.py --hits ../level9_ops/l9_v2_checkpoint_hits.pkl --episodes 3600 --ent-coef 0.15
"""

import argparse, pickle, signal, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l10 import IbexL10Env, MODULES, N_OBS

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CKPT_MODEL   = THIS / "l10_checkpoint_model"
CKPT_HITS    = THIS / "l10_checkpoint_hits.pkl"
CKPT_HISTORY = THIS / "l10_checkpoint_history.npz"

# ── Ctrl+C handler ────────────────────────────────────────────────────────────
_stop_requested = False

def _sigint_handler(sig, frame):
    global _stop_requested
    print("\n[!] Stop cerut — salvez la finalul episodului curent...", flush=True)
    _stop_requested = True

signal.signal(signal.SIGINT, _sigint_handler)


# ── Callback ──────────────────────────────────────────────────────────────────

class Log(BaseCallback):
    def __init__(self, baseline_pct: float, checkpoint_every: int = 100):
        super().__init__()
        self.baseline_pct     = baseline_pct
        self.checkpoint_every = checkpoint_every
        self.history = []

    def _on_step(self):
        global _stop_requested
        info = self.locals.get("infos", [{}])[0]
        if "cum_pct" not in info:
            return True

        ep      = len(self.history) + 1
        cum     = info["cum_pct"]
        new     = info.get("new_hits_vs_cum", 0)
        worst   = info.get("worst_mod", "?")
        worst_p = info.get("worst_pct", 0.0)
        self.history.append({
            "ep": ep, "cum_pct": cum,
            "ep_pct": info["ep_pct"], "new_hits": new,
        })
        delta = cum - self.baseline_pct
        print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | cum {cum:>5.2f}% | "
              f"new {new:>4} | Δ {delta:>+6.2f}pp | worst: {worst} {worst_p:.1f}%",
              flush=True)

        if ep % self.checkpoint_every == 0:
            self._save(ep, reason="checkpoint")

        if _stop_requested:
            self._save(ep, reason="interrupt")
            return False

        return True

    def _save(self, ep: int, reason: str = "checkpoint"):
        self.model.save(str(CKPT_MODEL))

        env = self.training_env.envs[0]
        if hasattr(env, "env"):
            env = env.env
        with open(CKPT_HITS, "wb") as f:
            pickle.dump(env._cum_hits, f)

        if self.history:
            eps    = np.array([h["ep"]      for h in self.history])
            cum    = np.array([h["cum_pct"] for h in self.history])
            ep_pct = np.array([h["ep_pct"]  for h in self.history])
            np.savez(CKPT_HISTORY, ep=eps, cum_pct=cum, ep_pct=ep_pct)

        print(f"  [{reason}] ep={ep} salvat → {CKPT_MODEL}.zip "
              f"({len(env._cum_hits):,} hits)", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3600)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="l10_ppo_curve.npz")
    ap.add_argument("--resume",   action="store_true",
                    help="Continuă din ultimul checkpoint L10")
    ap.add_argument("--hits",     default=None,
                    help="Fișier .pkl cu hits pre-acumulate (ex: din L9)")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="Suprascrie ent_coef (ex: 0.15 pentru mai multă explorare)")
    args = ap.parse_args()

    print("=" * 64)
    print(f"L10 PPO — obs {N_OBS} dims, {args.steps} pași/episod, 87 ops")
    print("=" * 64)

    initial_hits  = set()
    episodes_done = 0
    history_prev  = []

    if args.hits and Path(args.hits).exists():
        with open(args.hits, "rb") as f:
            initial_hits = pickle.load(f)
        print(f"  Hits pre-încărcate din {args.hits}: {len(initial_hits):,}")

    if args.resume and CKPT_HITS.exists() and CKPT_MODEL.with_suffix(".zip").exists():
        with open(CKPT_HITS, "rb") as f:
            initial_hits = pickle.load(f)
        if CKPT_HISTORY.exists():
            d = np.load(CKPT_HISTORY)
            episodes_done = int(d["ep"][-1])
            for i in range(len(d["ep"])):
                history_prev.append({
                    "ep":      int(d["ep"][i]),
                    "cum_pct": float(d["cum_pct"][i]),
                    "ep_pct":  float(d["ep_pct"][i]),
                    "new_hits": 0,
                })
        print(f"  Resume din ep {episodes_done} — {len(initial_hits):,} hits pre-încărcate")
    else:
        if args.resume:
            print("  [!] Niciun checkpoint L10 găsit — pornesc de la zero")

    remaining = args.episodes - episodes_done
    if remaining <= 0:
        print(f"  Deja {episodes_done} episoade făcute, nimic de rulat."); return

    baseline_pct = 100. * len(initial_hits) / 20248 if initial_hits else 0.0

    print(f"  Episoade totale: {args.episodes}  (rămase: {remaining})")
    print(f"  Pași/episod:     {args.steps}")
    print(f"  Obs dims:        {N_OBS}")
    ent_coef = args.ent_coef if args.ent_coef is not None else 0.08
    print(f"  Net arch:        [512, 256]  ent_coef={ent_coef}")
    print(f"  Baseline (L9):   {baseline_pct:.2f}%")
    print(f"  Checkpoint:      la fiecare {args.checkpoint_every} ep → {CKPT_MODEL}.zip")
    print(f"\n{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | {'Δbaseline':>10} | worst module")
    print("-" * 72)

    env = IbexL10Env(
        episode_steps=args.steps,
        seed=args.seed,
        initial_hits=initial_hits,
    )

    if args.resume and CKPT_MODEL.with_suffix(".zip").exists():
        model = PPO.load(str(CKPT_MODEL), env=env)
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef
        print(f"  Model încărcat din {CKPT_MODEL}.zip")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=args.steps,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=[512, 256]),
            verbose=0, seed=args.seed, device="cpu",
        )

    cb = Log(baseline_pct=baseline_pct, checkpoint_every=args.checkpoint_every)
    cb.history = history_prev

    t0 = time.time()
    model.learn(total_timesteps=remaining * args.steps, callback=cb)
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed/60:.1f} min")

    all_history = cb.history
    if all_history:
        best  = max(h["cum_pct"] for h in all_history)
        final = all_history[-1]["cum_pct"]
        print(f"\nRezultate finale:")
        print(f"  L10 PPO best:  {best:.2f}%")
        print(f"  L10 PPO final: {final:.2f}%")
        print(f"  L9 ceiling:   ~71.97%  (referință)")
        print(f"  L8 ceiling:   ~74.72%  (referință)")

        eps    = np.array([h["ep"]      for h in all_history])
        cum    = np.array([h["cum_pct"] for h in all_history])
        ep_pct = np.array([h["ep_pct"]  for h in all_history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
