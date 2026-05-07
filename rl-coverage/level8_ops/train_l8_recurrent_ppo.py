"""train_l8_recurrent_ppo.py — RecurrentPPO (LSTM) cu codec_l8_2 (72 ops).

Față de toate variantele anterioare de PPO:
  - Politică LSTM (MlpLstmPolicy) — are memorie între pași în interiorul episodului
  - Codec L8.2 (72 ops): include MRET și WFI față de L8 (70 ops)
  - Obs extins cu ultimele 4 op-uri executate (vezi env_l8_recurrent.py)

De ce LSTM ajută:
  - Secvențe critice (ECALL → ... → MRET) necesită că agentul să-și amintească
    ce a executat anterior în același episod
  - MLP-ul clasic decide fiecare acțiune independent — nu poate învăța secvențe

De ce N recent ops în obs ajută LSTM-ul:
  - Semnalul explicit pe termen scurt (ultimii 4 pași) reduce sarcina pe hidden state
  - Gradienții pentru dependențe scurte nu mai trebuie să traverseze tot LSTM-ul

Usage:
    python train_l8_recurrent_ppo.py
    python train_l8_recurrent_ppo.py --episodes 600 --steps 256
    python train_l8_recurrent_ppo.py --episodes 600 --hits l8_pipeline_hits.pkl
"""

import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

from env_l8_recurrent import IbexL8RecurrentEnv, TRACKED_MODULES

try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("[ERROR] pip install sb3-contrib stable-baselines3")
    sys.exit(1)


class Log(BaseCallback):
    def __init__(self, baseline_pct: float):
        super().__init__()
        self.baseline_pct = baseline_pct
        self.history = []

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "cum_pct" not in info:
            return True
        ep      = len(self.history) + 1
        cum     = info["cum_pct"]
        new     = info.get("new_hits_vs_cum", 0)
        worst   = info.get("worst_mod", "?")
        worst_p = info.get("worst_pct", 0.0)
        rew     = info.get("shaped_reward", 0.0)
        self.history.append({
            "ep": ep, "cum_pct": cum,
            "ep_pct": info["ep_pct"], "new_hits": new,
        })
        delta = cum - self.baseline_pct
        print(
            f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | cum {cum:>5.2f}% | "
            f"new {new:>4} | Δ {delta:>+6.2f}pp | rew {rew:>7.1f} | "
            f"worst: {worst} {worst_p:.1f}%",
            flush=True,
        )
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600,
                    help="Număr episoade de antrenament")
    ap.add_argument("--steps",    type=int, default=256,
                    help="Pași per episod (256 → mai multe episoade, mai rapid)")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--hits",     default=None,
                    help="Fișier .pkl cu hits pre-acumulate (ex: l8_pipeline_hits.pkl)")
    ap.add_argument("--out",      default="l8_recurrent_ppo_curve.npz")
    ap.add_argument("--lstm-size", type=int, default=256,
                    help="Dimensiunea hidden state LSTM")
    ap.add_argument("--lr",       type=float, default=3e-4)
    args = ap.parse_args()

    print("=" * 68)
    print("L8 RecurrentPPO — LSTM policy + codec_l8_2 (72 ops) + N recent ops")
    print("=" * 68)

    # Încarcă hits pre-acumulate (opțional)
    initial_hits = set()
    if args.hits and Path(args.hits).exists():
        with open(args.hits, "rb") as f:
            initial_hits = pickle.load(f)
        print(f"\n  Hits pre-încărcate: {len(initial_hits):,} din {args.hits}")
    else:
        if args.hits:
            print(f"\n  [WARN] {args.hits} nu există — pornire de la 0")
        else:
            print("\n  Pornire de la 0 (fără hits pre-acumulate)")

    env = IbexL8RecurrentEnv(
        episode_steps=args.steps,
        seed=args.seed,
        initial_hits=initial_hits,
    )

    # Baseline pct (din hits inițiale)
    from directed_l8 import run_raw
    s = run_raw([0x00000013] * 8)
    if s:
        total_tog    = s.by_kind["toggle"][1]
        baseline_pct = 100.0 * len(initial_hits) / max(total_tog, 1)
        env._total_tog = total_tog
    else:
        baseline_pct = 0.0
        print("  [WARN] Nu s-a putut obține total_tog — baseline_pct = 0")

    print(f"\n  Baseline:   {baseline_pct:.2f}%")
    print(f"  Codec:      L8.2 (72 ops — include MRET, WFI)")
    print(f"  Obs:        {env.observation_space.shape[0]} features "
          f"(2 global + 8 module + 4 recent ops)")
    print(f"  LSTM size:  {args.lstm_size}")
    print(f"  Episodes:   {args.episodes} × {args.steps} pași")
    print()

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.steps,          # un rollout = un episod
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        ent_coef=0.05,
        policy_kwargs=dict(
            lstm_hidden_size=args.lstm_size,
            n_lstm_layers=1,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
        verbose=0,
        seed=args.seed,
        device="cpu",
    )

    print(f"{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'new':>5} | "
          f"{'Δbaseline':>10} | {'reward':>8} | worst module")
    print("-" * 75)

    cb = Log(baseline_pct)
    t0 = time.time()
    model.learn(
        total_timesteps=args.episodes * args.steps,
        callback=cb,
    )
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed / 60:.1f} min")

    if cb.history:
        eps    = np.array([h["ep"]      for h in cb.history])
        cum    = np.array([h["cum_pct"] for h in cb.history])
        ep_pct = np.array([h["ep_pct"]  for h in cb.history])
        new_h  = np.array([h["new_hits"]for h in cb.history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct, new_hits=new_h)

        best  = cum.max()
        final = cum[-1]
        print(f"\nRezultate:")
        print(f"  Baseline:             {baseline_pct:.2f}%")
        print(f"  RecurrentPPO best:    {best:.2f}%")
        print(f"  RecurrentPPO final:   {final:.2f}%")
        print(f"  Câștig vs baseline:  {final - baseline_pct:+.2f} pp")
        print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
