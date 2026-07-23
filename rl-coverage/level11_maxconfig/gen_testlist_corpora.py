"""gen_testlist_corpora.py — generate corpus JSON(s) per portable profile in
testlist_l11.py (the Verilator-feasible subset of lowRISC's 57-test
testlist.yaml), for cpu/run_testlist_suite.sh to run against the vendored
opentitan-upstream build.

lowRISC's own testlist.yaml gives most tests an `iterations` count (e.g.
riscv_pmp_full_random_test: 600) — each iteration is a fresh random seed, run
independently. By default this script mirrors that via
testlist_l11.scaled_seed_counts(): upstream's real per-profile iteration counts,
scaled down to ~150 total runs (see that function's docstring for why — raw
upstream counts for just the 16 profiles here would mean ~1026 runs, ~870 of
them for PMP alone). --n-seeds overrides this with a flat count per profile
instead (the old behavior), for quick one-off experiments. Deterministic
profiles (currently just riscv_csr_test, a fixed sweep) are always generated
exactly once regardless of either mode, since re-running the same
seed-independent program is a no-op.

Usage:
    python3 gen_testlist_corpora.py                    # (default) scaled-upstream budget, ~150 total runs
    python3 gen_testlist_corpora.py --target-total 300  # scaled-upstream budget, ~300 total runs
    python3 gen_testlist_corpora.py --n-seeds 5          # flat override: 5 seeds/profile

Writes corpus_suite_<name>_seed<k>.json next to this file for every
(profile, seed) pair, and prints the SKIPPED list so it's visible at
generation time too (not just buried in testlist_l11.py).
"""
import argparse
import glob
import json
import os

from testlist_l11 import PROFILES, SKIPPED, scaled_seed_counts

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="first seed (subsequent seeds increment from here)")
    ap.add_argument("--n-seeds", type=int, default=None,
                     help="flat seed count per non-deterministic profile, overriding the "
                          "scaled-upstream budget below")
    ap.add_argument("--target-total", type=int, default=150,
                     help="total run budget for scaled_seed_counts() when --n-seeds is not given")
    ap.add_argument("--clean", action="store_true",
                     help="remove existing corpus_suite_*.json before generating")
    args = ap.parse_args()

    if args.clean:
        for stale in glob.glob(os.path.join(_HERE, "corpus_suite_*.json")):
            os.remove(stale)
        print(f"Removed existing corpus_suite_*.json")

    if args.n_seeds is not None:
        seed_counts = {p.name: (1 if p.deterministic else args.n_seeds) for p in PROFILES}
    else:
        seed_counts = scaled_seed_counts(target_total=args.target_total)
    print(f"Seed budget per profile (total {sum(seed_counts.values())} runs): "
          f"{seed_counts}\n")

    n_written = 0
    for p in PROFILES:
        n_seeds = seed_counts[p.name]
        seeds = [args.seed] if p.deterministic else list(range(args.seed, args.seed + n_seeds))
        for seed in seeds:
            machine_code = p.builder(seed)
            out_path = os.path.join(_HERE, f"corpus_suite_{p.name}_seed{seed}.json")
            payload = {
                "n": len(machine_code),
                "n_actions": p.n_actions,
                "agent": "testlist_suite_v1",
                "test_name": p.name,
                "description": p.description,
                "seed": seed,
                "category_weights": p.category_weights,
                "machine_code": machine_code,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f)
            n_written += 1
        print(f"Wrote {len(seeds)} seed(s) for {p.name} ({len(machine_code)} words each)")

    print(f"\n{n_written} corpora written across {len(PROFILES)} profiles. "
          f"{len(SKIPPED)} lowRISC tests skipped:")
    for name, reason in SKIPPED:
        print(f"  [SKIP] {name}: {reason}")


if __name__ == "__main__":
    main()
