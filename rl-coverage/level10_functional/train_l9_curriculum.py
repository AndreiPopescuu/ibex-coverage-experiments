"""train_l9_curriculum.py — PPO cu Curriculum Learning (versiunea fixată).

Curriculum:
  Ep 1-75:    128 pași
  Ep 76-150:  256 pași
  Ep 151-225: 512 pași
  Ep 226-300: 1024 pași

Total instrucțiuni: 75×128 + 75×256 + 75×512 + 75×1024 = 144.000

Logica corectă:
  - La fiecare episod, agentul generează EXACT current_steps acțiuni
  - PPO se antrenează pe acele acțiuni (nu folosim model.learn intern)
  - Când steps cresc, PPO primește rollout-uri mai lungi → credit assignment mai bun

Usage:
    python train_l9_curriculum.py --no-directed
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from env_l9 import IbexL9Env

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


CURRICULUM = [
    (1,   128),
    (76,  256),
    (151, 512),
    (226, 1024),
]
TOTAL_EPISODES = 300
TOTAL_INSTRS   = 75*128 + 75*256 + 75*512 + 75*1024


def get_steps(ep: int) -> int:
    steps = CURRICULUM[0][1]
    for ep_start, s in CURRICULUM:
        if ep >= ep_start:
            steps = s
    return steps


def make_ppo(env, n_steps, ent_coef=0.15, old_params=None):
    """Creează un PPO nou cu n_steps corect, opțional restaurând weights."""
    model = PPO("MlpPolicy", env,
        n_steps       = n_steps,
        batch_size    = min(64, n_steps),
        n_epochs      = 8,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        learning_rate = 3e-4,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-directed", action="store_true")
    ap.add_argument("--driver", default="test_run_for_l9")
    args = ap.parse_args()

    print("=" * 60)
    print("PPO cu CURRICULUM LEARNING")
    print(f"  Ep 1-75:    128 pași")
    print(f"  Ep 76-150:  256 pași")
    print(f"  Ep 151-225: 512 pași")
    print(f"  Ep 226-300: 1024 pași")
    print(f"  Total instrucțiuni: {TOTAL_INSTRS:,}")
    print("=" * 60)

    if not args.no_directed:
        try:
            from directed_l8 import run_directed
            print("\nDirected seed..."); run_directed()
        except Exception as e:
            print(f"[WARN] {e}")

    # Pornim cu 128 pași
    current_steps = 128
    env = IbexL9Env(episode_steps=current_steps, driver=args.driver)
    model = make_ppo(env, n_steps=current_steps, ent_coef=0.15)

    ep_pcts = []; cum_pcts = []; new_hits_log = []; steps_log = []

    print(f"\n{'ep':>5}  {'steps':>6}  {'ep%':>7}  {'cum%':>8}  {'new':>6}  {'time':>6}")
    print("-" * 52)

    t_total = time.time()

    for ep in range(1, TOTAL_EPISODES + 1):
        new_steps = get_steps(ep)

        # ── Tranziție curriculum ──────────────────────────────────────────
        if new_steps != current_steps:
            print(f"\n  ── curriculum: {current_steps} → {new_steps} pași/ep ──\n")
            old_params    = model.policy.state_dict()
            current_steps = new_steps
            env.episode_steps = current_steps
            # Entropy scade pe măsură ce steps cresc (exploatare mai multă)
            ent = max(0.05, 0.15 - (ep / TOTAL_EPISODES) * 0.10)
            model = make_ppo(env, n_steps=current_steps,
                             ent_coef=ent, old_params=old_params)

        # ── Colectare rollout (exact current_steps acțiuni) ───────────────
        t0 = time.time()

        # Antrenează PPO pentru exact un rollout de current_steps pași
        model.learn(
            total_timesteps     = current_steps,
            reset_num_timesteps = (ep == 1),
        )

        # Extrage info din ultimul episod completat
        ep_pct  = 0.0; cum_pct = 0.0; new_n = 0
        # Citim din env direct după learn (learn rulează env intern)
        # Alternativ rulăm un episod de evaluare
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, _, done, info = env.step(action)

        ep_pct  = info.get("ep_pct",  0.0)
        cum_pct = info.get("cum_pct", 0.0)
        new_n   = info.get("new_hits", 0)
        elapsed = int(time.time() - t0)

        ep_pcts.append(ep_pct)
        cum_pcts.append(cum_pct)
        new_hits_log.append(new_n)
        steps_log.append(current_steps)

        print(f"{ep:>5}  {current_steps:>6}  {ep_pct:>6.1f}%  "
              f"{cum_pct:>7.1f}%  {new_n:>6}  {elapsed:>5}s")

    print("=" * 60)
    print(f"Final cum coverage: {cum_pcts[-1]:.2f}%")
    print(f"Total instrucțiuni: {TOTAL_INSTRS:,}")
    print(f"Wall time: {time.time()-t_total:.0f}s")

    save = THIS / "l9_ppo_curriculum_curve.npz"
    np.savez(str(save),
        ep_pct   = np.array(ep_pcts),
        cum_pct  = np.array(cum_pcts),
        new_hits = np.array(new_hits_log),
        steps    = np.array(steps_log),
    )
    print(f"[saved] {save}")

    model.save(str(THIS / "l9_ppo_curriculum"))
    print(f"[saved] {THIS / 'l9_ppo_curriculum'}.zip")


if __name__ == "__main__":
    main()
