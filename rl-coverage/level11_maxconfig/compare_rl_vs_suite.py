"""compare_rl_vs_suite.py — compares the toggle bins found by RL training
(a checkpoint's _cum_hits pickle) against the union found by the 171-test
testlist-suite (cpu/coverage_suite_*.dat), to see whether RL's remaining gap
is concentrated in specific modules (e.g. PMP/CSR/illegal-instr, the ones
the suite's profiles specifically targeted) or spread out generally.

Usage (from rl-coverage/level11_maxconfig/, after RL training has produced
at least one checkpoint):
    python3 compare_rl_vs_suite.py [path/to/checkpoint_hits.pkl]

Defaults to l11_opentitan_upstream_checkpoint_hits.pkl if no path is given.
"""
import sys
import glob
import pickle
from pathlib import Path
from collections import Counter

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
sys.path.insert(0, str(THIS.parent / "level10_ops"))

import cov_parser
from env_l11 import _norm_key, _module_of

TOTAL_BINS = 38696


def main():
    hits_file = sys.argv[1] if len(sys.argv) > 1 else \
        str(THIS / "l11_opentitan_upstream_checkpoint_hits.pkl")

    cpu_dir = (THIS.parent.parent / "cpu").resolve()
    dat_paths = sorted(glob.glob(str(cpu_dir / "coverage_suite_*.dat")))
    if not dat_paths:
        print("ERROR: no coverage_suite_*.dat found in", cpu_dir)
        print("(gitignored, generated locally by cpu/run_testlist_suite.sh — "
              "make sure the 171-test suite ran on this machine)")
        sys.exit(1)

    suite_hits = set()
    for path in dat_paths:
        summary = cov_parser.parse(path)
        raw = cov_parser.hit_set(summary, cov_parser.KIND_TOGGLE)
        suite_hits |= {_norm_key(k) for k in raw}

    print(f"Suite (171 teste): {len(suite_hits)} bin-uri toggle unice, "
          f"din {len(dat_paths)} fisiere .dat")

    with open(hits_file, "rb") as f:
        rl_hits = pickle.load(f)
    print(f"RL ({Path(hits_file).name}): {len(rl_hits)} bin-uri toggle")

    both = suite_hits & rl_hits
    suite_only = suite_hits - rl_hits
    rl_only = rl_hits - suite_hits
    union = suite_hits | rl_hits

    print()
    print(f"Comune (ambele metode):        {len(both)}")
    print(f"Doar in suita (RL NU a gasit): {len(suite_only)}")
    print(f"Doar in RL (suita NU a gasit): {len(rl_only)}")
    print()
    pct = 100.0 * len(union) / TOTAL_BINS
    print(f"Uniune totala (suita + RL): {len(union)} / {TOTAL_BINS} "
          f"= {pct:.2f}%")

    mod_counts = Counter()
    for k in suite_only:
        mod = _module_of(k)
        mod_counts[mod or "?"] += 1

    print()
    print("Bin-uri gasite de suita dar RATATE de RL, pe modul (top 20):")
    for mod, cnt in mod_counts.most_common(20):
        print(f"  {mod:30s} {cnt}")

    print("\nDONE")


if __name__ == "__main__":
    main()
