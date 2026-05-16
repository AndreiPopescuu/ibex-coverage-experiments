"""train_l9_hybrid.py — PPO Shadow + Constrained Random hybrid.

Ideea: PPO singur platouiaza la ~77%. Constrained random adauga bins pe categorii
specifice (RAW hazards, operanzi extremi) pe care PPO nu le-a invatat inca.
Combinand ambeele pe acelasi _cum_hits, targetam >80%.

Strategia — interleaved execution:
  - La fiecare HYBRID_K episoade PPO, rulam 1 episod constrained random
  - Ambele contribuie la acelasi _cum_hits din shadow env
  - HYBRID_K creste in timp: incepe la 2 (50% constrained) → 20 (5% constrained)
  - Constrained tinteste bins ramase cu heuristici specifice (RAW, extreme ops)
  - PPO vede obs actualizate dupa fiecare constrained episode → isi ajusteaza politica

Usage:
    python train_l9_hybrid.py
    python train_l9_hybrid.py --episodes 4000
    python train_l9_hybrid.py --no-vtop
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from env_l9_shadow import IbexL9ShadowEnv, _TOTAL_BINS
from constrained_random_l9 import ConstrainedRandomAgent

try:
    from stable_baselines3 import PPO
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)


EPISODE_STEPS  = 480
TOTAL_EPISODES = 4000
VTOP_VALID_FREQ = 200
VTOP_VALID_N    = 3

# Constrained random frequency schedule:
#   hybrid_k(ep) = la ce N episoade PPO inseram 1 ep constrained random
#   Incepe la 2 (50% constrained), creste la 20 (5% constrained) dupa jumatate
HYBRID_K_START = 2
HYBRID_K_END   = 20
HYBRID_K_PIVOT = TOTAL_EPISODES // 2   # ep la care ajungem la K_END


def hybrid_k(ep: int) -> int:
    """Cate episoade PPO intre 2 episoade constrained random."""
    t = min(1.0, ep / HYBRID_K_PIVOT)
    k = HYBRID_K_START + t * (HYBRID_K_END - HYBRID_K_START)
    return max(HYBRID_K_START, int(k))


def make_ppo(env, n_steps, ent_coef=0.20, old_params=None):
    model = PPO(
        "MlpPolicy", env,
        n_steps       = n_steps,
        batch_size    = 96,
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


def run_constrained_episode(env: IbexL9ShadowEnv,
                             agent: ConstrainedRandomAgent) -> dict:
    """Ruleaza 1 episod constrained random pe shadow env (actualizeaza _cum_hits)."""
    agent.reset()
    actions = [agent.sample() for _ in range(env.episode_steps)]

    # Simulam direct prin shadow (bypass gym step loop pentru viteza)
    from shadow_riscv import RISCVShadow
    shadow = RISCVShadow()
    shadow.reset()
    result = shadow.run_actions(actions)

    ep_hits  = set(result["hits"])
    new_hits = ep_hits - env._cum_hits
    env._cum_hits |= ep_hits

    # Actualizeaza si contoarele de categorie din env (pentru obs corecte)
    env._update_cat_cache(new_hits)
    env._update_pair_caches(new_hits)
    env._n_episodes += 1

    cum_pct = 100.0 * len(env._cum_hits) / _TOTAL_BINS
    return {
        "ep_pct"  : result["pct"],
        "cum_pct" : cum_pct,
        "new_hits": len(new_hits),
    }


def vtop_validate(model, n_episodes: int, driver: str) -> float:
    try:
        from env_l9 import IbexL9Env
    except ImportError:
        return 0.0
    venv = IbexL9Env(episode_steps=EPISODE_STEPS, driver=driver)
    for _ in range(n_episodes):
        obs, _ = venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, done, info = venv.step(action)
    return info.get("cum_pct", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",  type=int,  default=TOTAL_EPISODES)
    ap.add_argument("--steps",     type=int,  default=EPISODE_STEPS)
    ap.add_argument("--no-vtop",   action="store_true")
    ap.add_argument("--driver",    default="test_run_for_l9")
    args = ap.parse_args()

    print("=" * 72)
    print("PPO SHADOW + CONSTRAINED RANDOM HYBRID")
    print(f"  {args.steps} pasi/ep, {args.episodes} episoade PPO")
    print(f"  Constrained freq: K={HYBRID_K_START}→{HYBRID_K_END} "
          f"(1 ep constrained la fiecare K ep PPO)")
    print(f"  Total bins (shadow): {_TOTAL_BINS}")
    print("=" * 72)

    env            = IbexL9ShadowEnv(episode_steps=args.steps,
                                      max_episodes=args.episodes)
    constrained    = ConstrainedRandomAgent(episode_steps=args.steps)
    model          = make_ppo(env, n_steps=args.steps, ent_coef=0.25)

    ep_pcts     = []; cum_pcts     = []
    new_hits_log= []; type_log     = []   # "ppo" sau "cr"
    vtop_log    = []
    best_vtop   = 0.0
    constrained_count = 0
    ppo_since_last_cr = 0

    t_total = time.time()
    print(f"\n{'ep':>6}  {'type':>5}  {'ep%':>7}  {'cum%':>8}  "
          f"{'new':>6}  {'rew':>8}  {'s':>5}")
    print("-" * 58)

    for ep in range(1, args.episodes + 1):
        k = hybrid_k(ep)
        run_cr = (ppo_since_last_cr >= k)

        t0 = time.time()

        if run_cr:
            # ── Episod constrained random ────────────────────────────────────
            info = run_constrained_episode(env, constrained)
            ep_pct  = info["ep_pct"]
            cum_pct = info["cum_pct"]
            new_n   = info["new_hits"]
            ep_rew  = float("nan")
            ep_type = "CR"
            ppo_since_last_cr = 0
            constrained_count += 1
        else:
            # ── Episod PPO ───────────────────────────────────────────────────
            model.learn(
                total_timesteps     = args.steps,
                reset_num_timesteps = (ep == 1),
            )
            obs, _ = env.reset()
            done   = False; ep_rew = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, rew, _, done, info = env.step(action)
                ep_rew += rew

            ep_pct  = info.get("ep_pct",  0.0)
            cum_pct = info.get("cum_pct", 0.0)
            new_n   = info.get("new_hits", 0)
            ep_type = "PPO"
            ppo_since_last_cr += 1

        elapsed = time.time() - t0
        ep_pcts.append(ep_pct)
        cum_pcts.append(cum_pct)
        new_hits_log.append(new_n)
        type_log.append(ep_type)

        # Checkpoint la fiecare 100 ep
        if ep % 100 == 0:
            avg_new = sum(new_hits_log[-100:]) / 100
            cr_frac = sum(1 for t in type_log[-100:] if t == "CR") / 100
            model.save(str(THIS / f"l9_hybrid_checkpoint_ep{ep}"))
            print(f"  ── ep {ep}: cum={cum_pct:.2f}%  avg_new={avg_new:.1f}  "
                  f"CR%={cr_frac*100:.0f}%  K={k}  [checkpoint] ──")

        # Verilator validation
        vtop_str = ""
        if not args.no_vtop and ep % VTOP_VALID_FREQ == 0:
            vtop_pct = vtop_validate(model, VTOP_VALID_N, args.driver)
            vtop_log.append((ep, vtop_pct))
            if vtop_pct > best_vtop:
                best_vtop = vtop_pct
                model.save(str(THIS / "l9_hybrid_best"))
                print(f"  [best vtop: {vtop_pct:.2f}%  saved]")
            vtop_str = f"  vtop={vtop_pct:.1f}%"

        rew_str = f"{ep_rew:>8.1f}" if ep_type == "PPO" else f"{'---':>8}"
        print(f"{ep:>6}  {ep_type:>5}  {ep_pct:>6.2f}%  {cum_pct:>7.2f}%  "
              f"{new_n:>6}  {rew_str}  {elapsed:>4.1f}s{vtop_str}")

    # ── Final report ──────────────────────────────────────────────────────────
    ppo_count = args.episodes - constrained_count
    print("=" * 72)
    print(f"Final shadow cum coverage: {cum_pcts[-1]:.2f}%")
    print(f"Episoade PPO: {ppo_count}  |  Episoade CR: {constrained_count}")
    print(f"Wall time: {time.time() - t_total:.0f}s")
    if vtop_log:
        print(f"Best Verilator: {best_vtop:.2f}%")

    save_path = THIS / "l9_hybrid_curve.npz"
    np.savez(str(save_path),
             ep_pct   = np.array(ep_pcts),
             cum_pct  = np.array(cum_pcts),
             new_hits = np.array(new_hits_log),
             type_log = np.array(type_log),
    )
    print(f"[saved] {save_path}")

    model.save(str(THIS / "l9_hybrid_final"))
    print(f"[saved] {THIS / 'l9_hybrid_final'}.zip")

    if vtop_log:
        vtop_arr = np.array(vtop_log)
        np.savez(str(THIS / "l9_hybrid_vtop_log.npz"),
                 ep=vtop_arr[:, 0], pct=vtop_arr[:, 1])


if __name__ == "__main__":
    main()
