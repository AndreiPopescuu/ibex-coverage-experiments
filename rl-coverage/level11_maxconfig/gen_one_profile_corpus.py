"""gen_one_profile_corpus.py — generate corpus_suite_<name>_seed*.json for a single
profile from testlist_l11.PROFILES, leaving every other profile's existing corpus
files untouched. Companion to gen_testlist_corpora.py, which always regenerates
every profile at once.

By default uses the same scaled_seed_counts() budget as the full suite. Pass
--target-instructions to instead pick a seed count that makes this profile's total
instruction count match some other total (e.g. the full 171-test suite's, for a
fair apples-to-apples coverage comparison against a single profile's own volume).

Usage:
    python3 gen_one_profile_corpus.py <profile_name>
    python3 gen_one_profile_corpus.py <profile_name> --n-seeds 20
    python3 gen_one_profile_corpus.py <profile_name> --target-instructions 139406
"""
import argparse
import json

from testlist_l11 import PROFILES, scaled_seed_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--n-seeds", type=int, default=None,
                     help="explicit seed count, overriding the scaled-upstream budget")
    ap.add_argument("--target-instructions", type=int, default=None,
                     help="pick a seed count so total instructions across all seeds "
                          "is as close as possible to this total (seed0's word count "
                          "is used to estimate seeds needed, since streams here are "
                          "fixed-length per profile)")
    args = ap.parse_args()

    p = next((p for p in PROFILES if p.name == args.profile), None)
    if p is None:
        valid = ", ".join(p.name for p in PROFILES)
        print(f"ERROR: no profile named {args.profile!r}. Valid names: {valid}")
        raise SystemExit(1)

    if p.deterministic:
        n_seeds = 1
    elif args.target_instructions is not None:
        n0 = len(p.builder(0))
        n_seeds = max(1, round(args.target_instructions / n0))
        print(f"seed0 = {n0} words -> {n_seeds} seed(s) to reach ~{args.target_instructions} "
              f"words (actual: {n_seeds * n0})")
    elif args.n_seeds is not None:
        n_seeds = args.n_seeds
    else:
        n_seeds = scaled_seed_counts()[p.name]

    for seed in range(n_seeds):
        machine_code = p.builder(seed)
        payload = {
            "n": len(machine_code), "n_actions": p.n_actions,
            "agent": "testlist_suite_v1", "test_name": p.name,
            "description": p.description, "seed": seed,
            "category_weights": p.category_weights, "machine_code": machine_code,
        }
        out = f"corpus_suite_{p.name}_seed{seed}.json"
        with open(out, "w") as f:
            json.dump(payload, f)
        print(f"  wrote {out} ({len(machine_code)} words)")
    print(f"{n_seeds} corpus file(s) written for {p.name}")


if __name__ == "__main__":
    main()
