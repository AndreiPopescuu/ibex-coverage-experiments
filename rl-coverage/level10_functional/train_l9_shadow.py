"""train_l9_shadow.py — PPO cu shadow simulator (fără Verilator).

Strategia:
  1. Shadow training: 5000 episoade cu IbexL9ShadowEnv (rapid)
     Curriculum: ep 1-1000 → 256 steps, ep 1001-2500 → 512 steps, ep 2501+ → 1024 steps
  2. Verilator validation: la fiecare 200 ep, rulează 3 episoade reale pentru calibrare
  3. Salvare model cel mai bun (după Verilator validation)

De ce shadow bate constrained_random:
  - 5000 ep vs 300 ep (16× mai multe episoade în același timp)
  - Obs space real (nu 3 câmpuri zero) → PPO știe ce categorii sunt descoperite
  - Curriculum mai lung → pași mai mulți → mai multe bins per episod

Usage:
    python train_l9_shadow.py
    python train_l9_shadow.py --no-vtop        # skip Verilator validation
    python train_l9_shadow.py --episodes 3000  # mai puține episoade
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from env_l9_shadow import IbexL9ShadowEnv, _TOTAL_BINS

try:
    from stable_baselines3 import PPO
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


CURRICULUM = [
    (1, 480),
]
TOTAL_EPISODES  = 5000
VTOP_VALID_FREQ = 200   # validare Verilator la fiecare N episoade
VTOP_VALID_N    = 3     # câte episoade Verilator pe rundă de validare


def get_steps(ep: int) -> int:
    steps = CURRICULUM[0][1]
    for ep_start, s in CURRICULUM:
        if ep >= ep_start:
            steps = s
    return steps


def make_ppo(env, n_steps, ent_coef=0.20, old_params=None):
    model = PPO(
        "MlpPolicy", env,
        n_steps       = n_steps,
        batch_size    = min(128, n_steps),
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        learning_rate = 2e-4,
        ent_coef      = ent_coef,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 0,
        device        = "cpu",
        policy_kwargs = dict(net_arch=[512, 512, 256]),
    )
    if old_params is not None:
        model.policy.load_state_dict(old_params)
    return model


def vtop_validate(model, n_episodes: int, driver: str) -> tuple[float, int]:
    """Rulează n_episodes cu modelul pe Verilator real. Returnează (cum_pct, new_hits)."""
    try:
        from env_l9 import IbexL9Env
    except ImportError:
        return 0.0, 0

    steps = get_steps(TOTAL_EPISODES)   # folosim pași maximi la validare
    venv  = IbexL9Env(episode_steps=steps, driver=driver)
    total_new = 0

    for _ in range(n_episodes):
        obs, _ = venv.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, done, info = venv.step(action)

    cum_pct = info.get("cum_pct", 0.0)
    return cum_pct, len(venv._cum_hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",  type=int,  default=TOTAL_EPISODES)
    ap.add_argument("--no-vtop",   action="store_true")
    ap.add_argument("--driver",    default="test_run_for_l9")
    ap.add_argument("--no-directed", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("PPO + SHADOW SIMULATOR (fără Verilator în training)")
    print(f"  480 pași/ep fix (fără curriculum)")
    print(f"  Total episoade: {args.episodes}")
    print(f"  Validare Verilator: {'dezactivat' if args.no_vtop else f'la fiecare {VTOP_VALID_FREQ} ep'}")
    print("=" * 70)

    if not args.no_directed:
        try:
            from directed_l8 import run_directed
            print("\nDirected seed..."); run_directed()
        except Exception as e:
            print(f"[WARN] directed: {e}")

    current_steps = get_steps(1)
    env    = IbexL9ShadowEnv(episode_steps=current_steps, max_episodes=args.episodes)
    model  = make_ppo(env, n_steps=current_steps, ent_coef=0.30)

    ep_pcts      = []; cum_pcts   = []; new_hits_log = []
    steps_log    = []; vtop_log   = []
    best_vtop    = 0.0
    t_total      = time.time()

    print(f"\n{'ep':>6}  {'ep%':>7}  {'cum%':>8}  {'new':>6}  {'rew':>8}  {'s':>5}")
    print("-" * 52)

    for ep in range(1, args.episodes + 1):
        new_steps = get_steps(ep)

        if new_steps != current_steps:
            print(f"\n  ── curriculum: {current_steps} → {new_steps} pași/ep ──\n")
            old_params    = model.policy.state_dict()
            current_steps = new_steps
            env.episode_steps = current_steps
            ent = max(0.10, 0.30 - (ep / args.episodes) * 0.20)
            model = make_ppo(env, n_steps=current_steps,
                             ent_coef=ent, old_params=old_params)

        t0 = time.time()

        # Train un rollout
        model.learn(
            total_timesteps     = current_steps,
            reset_num_timesteps = (ep == 1),
        )

        # Episod de evaluare pe shadow
        obs, _ = env.reset()
        done   = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, rew, _, done, info = env.step(action)
            ep_rew += rew

        ep_pct  = info.get("ep_pct",  0.0)
        cum_pct = info.get("cum_pct", 0.0)
        new_n   = info.get("new_hits", 0)
        elapsed = time.time() - t0

        ep_pcts.append(ep_pct); cum_pcts.append(cum_pct)
        new_hits_log.append(new_n); steps_log.append(current_steps)

        # Save periodic + summary la fiecare 100 ep
        if ep % 100 == 0:
            model.save(str(THIS / f"l9_shadow_checkpoint_ep{ep}"))
            avg_new = sum(new_hits_log[-100:]) / 100
            print(f"  ── ep {ep}: cum={cum_pct:.2f}%  avg_new/ep={avg_new:.2f}  "
                  f"[checkpoint salvat] ──")

        # Verilator validation
        vtop_pct_str = ""
        if not args.no_vtop and ep % VTOP_VALID_FREQ == 0:
            vtop_pct, _ = vtop_validate(model, VTOP_VALID_N, args.driver)
            vtop_log.append((ep, vtop_pct))
            if vtop_pct > best_vtop:
                best_vtop = vtop_pct
                model.save(str(THIS / "l9_shadow_best"))
                print(f"  [best vtop model saved: {vtop_pct:.2f}%]")
            vtop_pct_str = f"  vtop={vtop_pct:.1f}%"

        print(f"{ep:>6}  {ep_pct:>6.2f}%  {cum_pct:>7.2f}%  "
              f"{new_n:>6}  {ep_rew:>8.1f}  {elapsed:>4.1f}s{vtop_pct_str}")

    # ── Final report ─────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"Final shadow cum coverage: {cum_pcts[-1]:.2f}%")
    print(f"Total bins (shadow): {_TOTAL_BINS}")
    print(f"Wall time: {time.time() - t_total:.0f}s")
    if vtop_log:
        print(f"Best Verilator validation: {best_vtop:.2f}%")

    # Salvare curve
    save_path = THIS / "l9_shadow_curve.npz"
    np.savez(str(save_path),
             ep_pct    = np.array(ep_pcts),
             cum_pct   = np.array(cum_pcts),
             new_hits  = np.array(new_hits_log),
             steps     = np.array(steps_log),
    )
    print(f"[saved] {save_path}")

    model.save(str(THIS / "l9_shadow_final"))
    print(f"[saved] {THIS / 'l9_shadow_final'}.zip")

    if vtop_log:
        vtop_arr = np.array(vtop_log)
        np.savez(str(THIS / "l9_shadow_vtop_log.npz"),
                 ep=vtop_arr[:, 0], pct=vtop_arr[:, 1])
        print(f"[saved] {THIS / 'l9_shadow_vtop_log.npz'}")


if __name__ == "__main__":
    main()
