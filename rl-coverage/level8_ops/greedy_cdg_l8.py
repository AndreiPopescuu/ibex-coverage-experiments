"""greedy_cdg_l8.py — Automatic Coverage-Directed Generation pentru L8.

Algoritmul (complet automat, fără cunoaștere arhitecturală):

  Fiecare rundă:
    1. Generează K programe candidate:
         - 50% random pur
         - 50% mutații ale celui mai bun program din runda anterioară
    2. Rulează fiecare prin Vtop (~15s/program)
    3. Măsoară câte toggle points NOI a găsit fiecare candidat
    4. Adaugă TOATE hiturile noi la setul cumulativ
    5. Cel mai bun candidat (most new hits) devine seed-ul rundei viitoare

  Mutații posibile (alese aleator):
    - Înlocuiește random 10% din instrucțiuni cu altele random
    - Inserează instrucțiuni random în poziții random
    - Șterge instrucțiuni random
    - Swap între două instrucțiuni

Fără LLM, fără cunoaștere hardware. Pur explorare automată.

Usage:
    python greedy_cdg_l8.py --rounds 25 --candidates 5
    python greedy_cdg_l8.py --rounds 25 --candidates 5 --no-warmup
"""

import argparse, sys, time
from pathlib import Path
import numpy as np

THIS  = Path(__file__).resolve().parent
L5    = (THIS.parent / "level5_real_rtl").resolve()
ML4DV = (THIS.parent.parent / "cpu").resolve()
COVDAT = ML4DV / "coverage.dat"

sys.path.insert(0, str(L5))
import cov_parser

from env_l8 import IbexL8Env, run_program
from directed_l8 import run_raw, cum_hits_from, nop
from codec_l8 import N_OPS, IMM_BUCKETS, emit_program

# ── Generare și mutație programe ──────────────────────────────────────────

Action = tuple  # (op_i, rd, rs1, rs2, imm_bucket)

def random_action(rng) -> Action:
    return (int(rng.integers(N_OPS)), int(rng.integers(32)),
            int(rng.integers(32)),    int(rng.integers(32)),
            int(rng.integers(IMM_BUCKETS)))

def random_program(rng, n_steps: int) -> list[Action]:
    return [random_action(rng) for _ in range(n_steps)]

def mutate(program: list[Action], rng, mutation_rate: float = 0.10) -> list[Action]:
    """Mutează un program: replace / insert / delete / swap."""
    p = list(program)
    n = len(p)
    ops = rng.choice(["replace", "insert", "delete", "swap"],
                     p=[0.50, 0.20, 0.15, 0.15])

    if ops == "replace":
        # Înlocuiește ~mutation_rate% din instrucțiuni
        n_mut = max(1, int(n * mutation_rate))
        idxs = rng.choice(n, size=n_mut, replace=False)
        for i in idxs:
            p[i] = random_action(rng)

    elif ops == "insert":
        # Inserează câteva instrucțiuni noi
        n_ins = max(1, int(n * mutation_rate / 2))
        for _ in range(n_ins):
            pos = int(rng.integers(len(p) + 1))
            p.insert(pos, random_action(rng))

    elif ops == "delete" and len(p) > 10:
        # Șterge câteva instrucțiuni
        n_del = max(1, int(n * mutation_rate / 2))
        idxs = sorted(rng.choice(len(p), size=n_del, replace=False), reverse=True)
        for i in idxs:
            p.pop(i)

    elif ops == "swap":
        # Swap între două instrucțiuni random
        if len(p) >= 2:
            i, j = rng.choice(len(p), size=2, replace=False)
            p[i], p[j] = p[j], p[i]

    return p

# ── Runner ────────────────────────────────────────────────────────────────

def run_actions(actions: list[Action]):
    """Rulează o listă de acțiuni codec prin Vtop. Returnează summary sau None."""
    machine = emit_program(actions)
    return run_raw(list(machine))

def eval_candidate(actions: list[Action], cum_hits: set) -> tuple[int, set]:
    """Returnează (new_hits_count, hits_set) pentru un program."""
    s = run_actions(actions)
    if s is None:
        return 0, set()
    hits = cum_hits_from(s)
    new_hits = hits - cum_hits
    return len(new_hits), hits

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds",      type=int, default=25,
                    help="Număr de runde greedy")
    ap.add_argument("--candidates",  type=int, default=5,
                    help="Candidați per rundă (mai mulți = mai lent dar mai bun)")
    ap.add_argument("--steps",       type=int, default=512,
                    help="Instrucțiuni per program candidat")
    ap.add_argument("--seed",        type=int, default=7)
    ap.add_argument("--no-warmup",   action="store_true",
                    help="Sare peste random warm-up (mai rapid)")
    ap.add_argument("--out",         default="l8_cdg_curve.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 64)
    print("L8 Greedy CDG — Coverage-Directed Generation automat")
    print(f"  {args.rounds} runde × {args.candidates} candidați × {args.steps} pași")
    print("=" * 64)

    cum_hits  = set()
    total_tog = None

    # ── Warm-up random ────────────────────────────────────────────────────
    if not args.no_warmup:
        print("\n[Warm-up] 5 ep random × 1024 pași...")
        env = IbexL8Env(episode_steps=1024, seed=args.seed, reward_mode="novelty")
        rng_wup = np.random.default_rng(args.seed + 1)
        for ep in range(5):
            env.reset()
            for _ in range(1024):
                a = [rng_wup.integers(N_OPS), rng_wup.integers(32),
                     rng_wup.integers(32), rng_wup.integers(32),
                     rng_wup.integers(IMM_BUCKETS)]
                _, _, _, trunc, _ = env.step(a)
                if trunc: break
            s = cov_parser.parse(str(COVDAT))
            cum_hits |= cum_hits_from(s)
            if total_tog is None:
                total_tog = s.by_kind["toggle"][1]
            print(f"  ep {ep+1}: cum={100.*len(cum_hits)/total_tog:.2f}%", flush=True)
    else:
        # Rulează 1 NOP ca să obținem total_tog
        s = run_raw([0x00000013] * 8)
        if s:
            total_tog = s.by_kind["toggle"][1]
            cum_hits |= cum_hits_from(s)

    baseline_pct = 100. * len(cum_hits) / total_tog
    print(f"\n  Baseline după warm-up: {baseline_pct:.2f}%")
    print(f"  Ceiling L8:           ~74.72%")
    print(f"  Gap:                   {74.72 - baseline_pct:.2f} pp\n")

    # ── Greedy CDG ────────────────────────────────────────────────────────
    print(f"{'rnd':>4} | {'best_new':>8} | {'tot_new':>8} | {'cum%':>7} | {'Δbase':>8} | {'s':>5}")
    print("-" * 55)

    seed_program: list[Action] | None = None
    history = []
    t_total = time.time()

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        candidates = []

        for c in range(args.candidates):
            if seed_program is None or rng.random() < 0.5:
                # 50% random pur
                prog = random_program(rng, args.steps)
            else:
                # 50% mutație a seed-ului
                prog = mutate(seed_program, rng)
            candidates.append(prog)

        # Evaluează toți candidații
        best_new   = 0
        best_prog  = None
        round_hits = set()

        for prog in candidates:
            n_new, hits = eval_candidate(prog, cum_hits)
            round_hits |= hits
            if n_new > best_new:
                best_new  = n_new
                best_prog = prog

        # Adaugă toate hiturile din această rundă
        new_total = len(round_hits - cum_hits)
        cum_hits |= round_hits

        if best_prog is not None:
            seed_program = best_prog

        cum_pct = 100. * len(cum_hits) / total_tog
        dt = time.time() - t0

        history.append({"rnd": rnd, "best_new": best_new,
                         "tot_new": new_total, "cum_pct": cum_pct})

        print(f"{rnd:>4} | {best_new:>8} | {new_total:>8} | {cum_pct:>6.2f}% | "
              f"{cum_pct - baseline_pct:>+7.2f}pp | {dt:>4.0f}s", flush=True)

    elapsed = time.time() - t_total
    final_pct = 100. * len(cum_hits) / total_tog

    print(f"\nDone în {elapsed/60:.1f} min")
    print(f"\nRezultate:")
    print(f"  Baseline (random warm-up): {baseline_pct:.2f}%")
    print(f"  CDG final:                 {final_pct:.2f}%")
    print(f"  Câștig CDG:               {final_pct - baseline_pct:+.2f} pp")
    print(f"  Ceiling L8:               ~74.72%")

    rnds = np.array([h["rnd"]     for h in history])
    cums = np.array([h["cum_pct"] for h in history])
    news = np.array([h["tot_new"] for h in history])
    np.savez(args.out, rnd=rnds, cum_pct=cums, new_hits=news)
    print(f"  Saved → {args.out}")

    # Salvează hiturile acumulate pentru PPO
    import pickle
    hits_file = args.out.replace(".npz", "_hits.pkl")
    with open(hits_file, "wb") as f:
        pickle.dump(cum_hits, f)
    print(f"  Hits saved → {hits_file}")


if __name__ == "__main__":
    main()
