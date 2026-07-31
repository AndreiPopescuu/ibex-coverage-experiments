"""train_l11_ppo.py — PPO pe configuratia "max" (lockstep, ICache, PMP, RV32B
gated on), folosind codec_l11 (143 ops: L10 RV32IMC + 29 RV32B Zba/Zbb/Zbs
+ 27 RV32B zbp/zbc/zbe/zbf).

Build-ul max trebuie compilat o data, fara tracing:
    cd cpu && make CONFIG=max sim_build_max/Vtop

Baseline (hits acumulate din replay-ul corpus-ului minimal pe build-ul max):
    python replay_corpus.py --corpus ../corpus_all.json --save-hits l11_baseline_hits.pkl

Curriculum (3 faze cu action space progresiv):
    # Faza 1 — 45 ops: RV32I core + RV32M + branches + JAL + CSR (porneste din L10 corpus)
    python train_l11_ppo.py --phase 1 --hits l11_baseline_hits.pkl --episodes 1200 --steps 256

    # Faza 2 — 87 ops: + RV32C + LUI/AUIPC/JALR + exceptii (porneste din coverage faza 1)
    python train_l11_ppo.py --phase 2 --hits l11_p1_checkpoint_hits.pkl --episodes 1200 --steps 256

    # Faza 3 — 143 ops: full cu RV32B Zba/Zbb/Zbs + zbp/zbc/zbe/zbf (porneste din coverage faza 2)
    python train_l11_ppo.py --phase 3 --hits l11_p2_checkpoint_hits.pkl --episodes 1200 --steps 256

Usage (full, fara curriculum):
    python train_l11_ppo.py --hits l11_baseline_hits.pkl --episodes 3600 --steps 256
    python train_l11_ppo.py --resume --episodes 3600 --steps 256
"""

import argparse, pickle, signal, sys, time
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
sys.path.insert(0, str(THIS.parent / "level10_ops"))

from env_l11 import IbexL11Env, MODULES, N_OBS, PHASE_MAX_OPS, VTOP_BUILDS, DEFAULT_BUILD
from codec_l11 import N_OPS, N_CSR_BUCKETS

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError:
    print("[ERROR] pip install stable-baselines3"); sys.exit(1)

try:
    from sb3_contrib import RecurrentPPO
    _RECURRENT_AVAILABLE = True
except ImportError:
    _RECURRENT_AVAILABLE = False

try:
    from sb3_contrib import MaskablePPO
    _MASKABLE_AVAILABLE = True
except ImportError:
    _MASKABLE_AVAILABLE = False

# ── Checkpoint paths (per-build + per-phase, sau generic pentru rulari fara
# --phase) — separate pe build, fiindca hits/model dintr-un RTL nu au sens
# reincarcate peste alt RTL (module/bin-uri diferite: vezi VTOP_BUILDS in
# env_l11.py) ───────────────────────────────────────────────────────────────
def _ckpt_paths(phase: int | None, build: str):
    build_tag = "" if build == DEFAULT_BUILD else f"{build}_"
    tag = f"{build_tag}p{phase}_" if phase else build_tag
    return (
        THIS / f"l11_{tag}checkpoint_model",
        THIS / f"l11_{tag}checkpoint_hits.pkl",
        THIS / f"l11_{tag}checkpoint_history.npz",
    )

# Total bins toggle per build (masurat o data via replay_corpus.py / suite run).
# Folosit doar pentru afisarea baseline-ului inainte de primul episod;
# self._total_tog din env se actualizeaza la valoarea reala dupa primul episod.
TOTAL_BINS = {
    "max":                33624,
    "opentitan_upstream":  38696,  # = totalul din baseline-ul testlist-suite (171 teste, 83.32%)
}

# ── Ctrl+C handler ────────────────────────────────────────────────────────────
_stop_requested = False

def _sigint_handler(sig, frame):
    global _stop_requested
    print("\n[!] Stop cerut — salvez la finalul episodului curent...", flush=True)
    _stop_requested = True

signal.signal(signal.SIGINT, _sigint_handler)


# ── Callback ──────────────────────────────────────────────────────────────────

class Log(BaseCallback):
    def __init__(self, baseline_pct: float, checkpoint_every: int = 100,
                 corpus_path: str | None = None, ckpt_model=None,
                 ckpt_hits=None, ckpt_history=None):
        super().__init__()
        self.baseline_pct     = baseline_pct
        self.checkpoint_every = checkpoint_every
        self.corpus_path      = corpus_path
        self._ckpt_model      = ckpt_model
        self._ckpt_hits       = ckpt_hits
        self._ckpt_history    = ckpt_history
        self.history          = []
        self._corpus          = []

    def _on_step(self):
        global _stop_requested
        # Cu n_envs > 1, toate mediile sunt sincronizate (acelasi episode_steps),
        # deci pot termina episodul in aceeasi rundă — "infos" are o intrare per
        # mediu, dar doar cele care tocmai s-au truncat au "cum_pct". Procesăm
        # pe rand fiecare episod incheiat in aceasta runda.
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "cum_pct" not in info:
                continue

            ep      = len(self.history) + 1
            cum     = info["cum_pct"]
            new     = info.get("new_hits_vs_cum", 0)
            worst   = info.get("worst_mod", "?")
            worst_p = info.get("worst_pct", 0.0)
            self.history.append({
                "ep": ep, "cum_pct": cum,
                "ep_pct": info["ep_pct"], "new_hits": new,
            })
            new_br = info.get("new_branch_hits", 0)
            delta  = cum - self.baseline_pct
            print(f"  ep {ep:>4} | ep {info['ep_pct']:>5.2f}% | cum {cum:>5.2f}% | "
                  f"tog+{new:<4} br+{new_br:<3} | Δ {delta:>+6.2f}pp | worst: {worst} {worst_p:.1f}%",
                  flush=True)

            words = info.get("ep_words")
            if words and self.corpus_path:
                self._corpus.append({
                    "ep": ep, "words": words,
                    "new_hits": new, "cum_pct": round(cum, 3),
                })
                self._save_corpus(cum)

            if ep % self.checkpoint_every == 0:
                self._save(ep, reason="checkpoint")

        if _stop_requested:
            self._save(len(self.history), reason="interrupt")
            return False

        return True

    def _save_corpus(self, cum_pct: float):
        import json
        with open(self.corpus_path, "w") as f:
            json.dump({
                "total_programs": len(self._corpus),
                "final_cum_pct":  round(cum_pct, 3),
                "programs":       self._corpus,
            }, f, indent=2)

    def _load_corpus(self):
        import json
        from pathlib import Path
        if self.corpus_path and Path(self.corpus_path).exists():
            with open(self.corpus_path) as f:
                data = json.load(f)
            self._corpus = data.get("programs", [])

    def _save(self, ep: int, reason: str = "checkpoint"):
        self.model.save(str(self._ckpt_model))

        # get_attr merge peste toate workerii (functioneaza identic pentru
        # DummyVecEnv si SubprocVecEnv — spre deosebire de .envs[0], care nu
        # exista la SubprocVecEnv si oricum ar pierde hits-urile descoperite
        # de ceilalti workeri).
        hit_sets    = self.training_env.get_attr("_cum_hits")
        merged_hits = set().union(*hit_sets) if hit_sets else set()
        with open(self._ckpt_hits, "wb") as f:
            pickle.dump(merged_hits, f)

        if self.history:
            eps    = np.array([h["ep"]      for h in self.history])
            cum    = np.array([h["cum_pct"] for h in self.history])
            ep_pct = np.array([h["ep_pct"]  for h in self.history])
            np.savez(self._ckpt_history, ep=eps, cum_pct=cum, ep_pct=ep_pct)

        print(f"  [{reason}] ep={ep} salvat → {self._ckpt_model}.zip "
              f"({len(merged_hits):,} hits, merged din {len(hit_sets)} worker(i))",
              flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",    type=int, choices=[1, 2, 3], default=None,
                    help="Faza curriculum (1=45 ops, 2=87 ops, 3=143 ops). "
                         "Fara --phase → action space complet (143 ops).")
    ap.add_argument("--build", choices=list(VTOP_BUILDS), default=DEFAULT_BUILD,
                    help="Ce Vtop antrenezi: 'max' (vechi, 33624 bins) sau "
                         "'opentitan_upstream' (RTL vendorizat proaspat, preset "
                         "oficial lowRISC, 38696 bins — exact build-ul pe care a "
                         "rulat testlist-suite-ul de 171 teste, baseline 83.32%%).")
    ap.add_argument("--episodes", type=int, default=3600)
    ap.add_argument("--steps",    type=int, default=256)
    ap.add_argument("--timeout",  type=int, default=600,
                    help="Timeout per program subprocess (secunde). "
                         "Crește proporțional cu --steps (ex: 1024 steps → 600s)")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      default="l11_ppo_curve.npz")
    ap.add_argument("--resume",   action="store_true",
                    help="Continuă din ultimul checkpoint L11")
    ap.add_argument("--hits",     default=None,
                    help="Fișier .pkl cu hits pre-acumulate "
                         "(ex: l11_baseline_hits.pkl din replay_corpus.py)")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--n-envs", type=int, default=1,
                    help="Nr. de medii paralele (SubprocVecEnv, cate un Vtop subprocess "
                         "per worker). Fiecare worker are hits/coverage proprii — se "
                         "unesc doar la checkpoint save, nu se sincronizeaza live.")
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="Suprascrie ent_coef (ex: 0.15 pentru mai multă explorare)")
    ap.add_argument("--recurrent", action="store_true",
                    help="Folosește RecurrentPPO (LSTM) din sb3-contrib în loc de PPO (MLP). "
                         "Permite coordonarea secvențelor multi-pas (ex: PMP setup → access).")
    ap.add_argument("--action-mask", action="store_true",
                    help="Folosește MaskablePPO (sb3-contrib): maschează op-urile ale căror "
                         "module RTL țintă (OP_MODULES în env_l11.py) sunt deja saturate "
                         "(>= --mask-saturation), forțând agentul spre module slab acoperite. "
                         "Doar dimensiunea 'op' e mascată — rd/rs1/rs2/imm/csr rămân libere. "
                         "Nu poate fi combinat cu --recurrent.")
    ap.add_argument("--mask-saturation", type=float, default=0.97,
                    help="Prag de acoperire per modul peste care un op devine candidat la "
                         "mascare, dacă TOATE modulele lui țintă sunt peste prag (default 0.97)")
    ap.add_argument("--mask-min-unmasked-frac", type=float, default=0.15,
                    help="Fracție minimă de op-uri care trebuie să rămână nemascate; sub asta "
                         "masking-ul se dezactivează pentru acel pas (anti-starvare, default 0.15)")
    ap.add_argument("--pretrained-model", default=None,
                    help="Încarcă weights dintr-un model existent dar resetează hits (fresh coverage)")
    ap.add_argument("--corpus-out", default="corpus_l11.json",
                    help="Fișier JSON pentru episoadele cu bins noi (append dacă există)")
    args = ap.parse_args()

    if args.action_mask and args.recurrent:
        ap.error("--action-mask și --recurrent nu pot fi combinate "
                 "(sb3-contrib nu are un MaskableRecurrentPPO).")
    if args.action_mask and not _MASKABLE_AVAILABLE:
        ap.error("--action-mask cere sb3-contrib. Instalează cu: pip install sb3-contrib")

    max_ops = PHASE_MAX_OPS[args.phase] if args.phase else N_OPS
    CKPT_MODEL, CKPT_HITS, CKPT_HISTORY = _ckpt_paths(args.phase, args.build)
    total_bins = TOTAL_BINS[args.build]

    phase_label = f"faza {args.phase} ({max_ops} ops)" if args.phase else f"full ({N_OPS} ops)"
    print("=" * 70)
    print(f"L11 PPO — build={args.build!r} (lockstep/ICache/PMP/RV32B on), {phase_label}, "
          f"obs {N_OBS} dims, {args.steps} pași/ep")
    print(f"  Action space: [{max_ops} ops, 32 rd, 32 rs1, 32 rs2, 5 imm, {N_CSR_BUCKETS} csr_bucket]")
    print(f"  Reward: episode_base + toggle_shaped + 0.3× branch")
    print("=" * 70)

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
            print(f"  [!] Niciun checkpoint găsit pentru {phase_label} — pornesc de la zero")

    remaining = args.episodes - episodes_done
    if remaining <= 0:
        print(f"  Deja {episodes_done} episoade făcute, nimic de rulat."); return

    baseline_pct = 100. * len(initial_hits) / total_bins if initial_hits else 0.0

    print(f"  Episoade totale: {args.episodes}  (rămase: {remaining})")
    print(f"  Pași/episod:     {args.steps}")
    print(f"  Obs dims:        {N_OBS}")
    print(f"  Medii paralele:  {max(1, args.n_envs)} "
          f"({'SubprocVecEnv' if args.n_envs > 1 else 'DummyVecEnv'})")
    ent_coef = args.ent_coef if args.ent_coef is not None else 0.08
    use_recurrent = args.recurrent and _RECURRENT_AVAILABLE
    use_maskable  = args.action_mask and _MASKABLE_AVAILABLE
    if args.recurrent and not _RECURRENT_AVAILABLE:
        print("  [!] sb3-contrib nu e instalat — fallback la PPO (MLP). "
              "Instalează cu: pip install sb3-contrib")
    AlgoCls   = MaskablePPO if use_maskable else (RecurrentPPO if use_recurrent else PPO)
    algo_name = ("MaskablePPO" if use_maskable else
                 "RecurrentPPO (LSTM)" if use_recurrent else "PPO (MLP)")
    net_desc  = "LSTM-256 + [256,256]" if use_recurrent else "[512, 256]"
    print(f"  Algoritm:        {algo_name}")
    print(f"  Net arch:        {net_desc}  ent_coef={ent_coef}")
    if use_maskable:
        print(f"  Action masking:  ON (dim 'op', saturation>={args.mask_saturation}, "
              f"min unmasked frac={args.mask_min_unmasked_frac})")
    print(f"  Baseline:        {baseline_pct:.2f}%  (din {len(initial_hits):,} / ~{total_bins:,} bins)")
    print(f"  Checkpoint:      la fiecare {args.checkpoint_every} ep → {CKPT_MODEL}.zip")
    print(f"\n{'ep':>5} | {'ep%':>6} | {'cum%':>6} | {'tog+':>5} {'br+':>4} | {'Δbaseline':>10} | worst module")
    print("-" * 75)

    def _make_env(rank: int):
        def _init():
            return IbexL11Env(
                episode_steps=args.steps,
                seed=args.seed + rank,
                initial_hits=initial_hits,
                timeout=args.timeout,
                max_ops=max_ops,
                mask_saturation=args.mask_saturation,
                mask_min_unmasked_frac=args.mask_min_unmasked_frac,
                build=args.build,
            )
        return _init

    n_envs = max(1, args.n_envs)
    if n_envs > 1:
        # Fiecare worker porneste cu acelasi initial_hits (baseline-ul de la --hits
        # / --resume), apoi diverge in timpul antrenarii — nu se sincronizeaza live
        # intre workeri (ar cere IPC pe fiecare step si ar anula castigul de viteza).
        # Se unesc doar la fiecare checkpoint save (_save() foloseste get_attr).
        env = SubprocVecEnv([_make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([_make_env(0)])

    model = None

    if args.resume and CKPT_MODEL.with_suffix(".zip").exists():
        try:
            # n_steps=args.steps suprascrie valoarea salvată în checkpoint —
            # necesar pentru schedule-ul de lungime episod (rulezi acest
            # script de mai multe ori cu --steps crescând, fiecare cu
            # --resume; fără suprascriere, rollout buffer-ul ar rămâne la
            # lungimea din prima rulare, ignorând noul --steps).
            model = AlgoCls.load(str(CKPT_MODEL), env=env, device="cpu",
                                  n_steps=args.steps)
            if args.ent_coef is not None:
                model.ent_coef = args.ent_coef
            print(f"  Model încărcat din {CKPT_MODEL}.zip")
        except Exception as e:
            print(f"  [!] Checkpoint incompatibil ({e}) — model nou")
            model = None
    elif args.pretrained_model:
        src = Path(args.pretrained_model)
        if not src.suffix:
            src = src.with_suffix(".zip")
        if src.exists():
            try:
                model = AlgoCls.load(str(src.with_suffix("")), env=env, device="cpu",
                                      n_steps=args.steps)
                if args.ent_coef is not None:
                    model.ent_coef = args.ent_coef
                print(f"  Weights încărcate din {src} — hits resetate (fresh coverage)")
            except Exception as e:
                print(f"  [!] Pretrained model incompatibil ({e}) — model nou")
                model = None
        else:
            print(f"  [!] {src} nu există — model nou")

    if model is None:
        if use_recurrent:
            model = RecurrentPPO(
                "MlpLstmPolicy", env,
                learning_rate=3e-4,
                n_steps=args.steps,
                batch_size=64,
                n_epochs=4,
                gamma=0.999,
                ent_coef=ent_coef,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    lstm_hidden_size=256,
                    enable_critic_lstm=True,
                ),
                verbose=0, seed=args.seed, device="cpu",
            )
        elif use_maskable:
            model = MaskablePPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=args.steps,
                batch_size=64,
                n_epochs=4,
                gamma=0.999,
                ent_coef=ent_coef,
                # dict form -> separate pi/vf trunks (not a shared [512,256]
                # backbone): the reward here can spike into the thousands on
                # early episodes (dynamic per-module weighting), and a shared
                # trunk lets that value-loss spike corrupt the policy's
                # features via backprop, collapsing it to a near-deterministic
                # output. Matches RecurrentPPO below, which already does this.
                policy_kwargs=dict(net_arch=dict(pi=[512, 256], vf=[512, 256])),
                verbose=0, seed=args.seed, device="cpu",
            )
        else:
            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=args.steps,
                batch_size=64,
                n_epochs=4,
                gamma=0.999,
                ent_coef=ent_coef,
                policy_kwargs=dict(net_arch=dict(pi=[512, 256], vf=[512, 256])),
                verbose=0, seed=args.seed, device="cpu",
            )

    cb = Log(baseline_pct=baseline_pct, checkpoint_every=args.checkpoint_every,
             corpus_path=str(THIS / args.corpus_out),
             ckpt_model=CKPT_MODEL, ckpt_hits=CKPT_HITS, ckpt_history=CKPT_HISTORY)
    cb._load_corpus()
    cb.history = history_prev

    t0 = time.time()
    model.learn(total_timesteps=remaining * args.steps, callback=cb)
    elapsed = time.time() - t0

    print(f"\nDone în {elapsed/60:.1f} min")

    # Final cross-worker merge: checkpoints only happen every
    # --checkpoint-every episodes, so a run that stops between two of them
    # (the common case, e.g. Ctrl+C or the curriculum wrapper's episode
    # target) would otherwise never report the true global union — only
    # cb.history's per-worker-local cum_pct (info["cum_pct"], scoped to
    # whichever single worker logged that episode; see env_l11.py's
    # _cum_hits — never synced live across workers, only merged here/at
    # checkpoints).
    cb._save(len(cb.history), reason="final")
    hit_sets    = env.get_attr("_cum_hits")
    merged_hits = set().union(*hit_sets) if hit_sets else set()
    tot_sets    = env.get_attr("_total_tog")
    real_total  = tot_sets[0] if tot_sets else total_bins
    global_pct  = 100.0 * len(merged_hits) / max(real_total, 1)

    all_history = cb.history
    print(f"\nRezultate finale:")
    print(f"  Uniune globală (toți {max(1, args.n_envs)} workerii): "
          f"{global_pct:.2f}%  ({len(merged_hits):,} / {real_total:,} bins)")
    if all_history:
        best = max(h["cum_pct"] for h in all_history)
        print(f"  Cel mai bun worker individual (local):    {best:.2f}%")
    print(f"  Baseline:                                  {baseline_pct:.2f}%")

    if all_history:
        eps    = np.array([h["ep"]      for h in all_history])
        cum    = np.array([h["cum_pct"] for h in all_history])
        ep_pct = np.array([h["ep_pct"]  for h in all_history])
        np.savez(args.out, ep=eps, cum_pct=cum, ep_pct=ep_pct)
        print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
