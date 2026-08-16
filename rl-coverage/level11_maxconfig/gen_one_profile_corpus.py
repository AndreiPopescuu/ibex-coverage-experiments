"""gen_one_profile_corpus.py — generate corpus_suite_<name>_seed*.json for a single
profile from testlist_l11.PROFILES, leaving every other profile's existing corpus
files untouched. Companion to gen_testlist_corpora.py, which always regenerates
every profile at once.

Usage:
    python3 gen_one_profile_corpus.py <profile_name>
"""
import json
import sys

from testlist_l11 import PROFILES, scaled_seed_counts


def main():
    if len(sys.argv) != 2:
        valid = ", ".join(p.name for p in PROFILES)
        print(f"Usage: python3 gen_one_profile_corpus.py <profile_name>\nValid names: {valid}")
        sys.exit(1)

    name = sys.argv[1]
    p = next((p for p in PROFILES if p.name == name), None)
    if p is None:
        valid = ", ".join(p.name for p in PROFILES)
        print(f"ERROR: no profile named {name!r}. Valid names: {valid}")
        sys.exit(1)

    n_seeds = 1 if p.deterministic else scaled_seed_counts()[p.name]
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
