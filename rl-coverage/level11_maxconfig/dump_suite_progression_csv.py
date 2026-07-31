"""dump_suite_progression_csv.py — exports the testlist-suite's cumulative
toggle-coverage progression, test by test, in the same run order as
cpu/run_testlist_suite.sh (sorted glob) — so it can be plotted on the same
x-axis (instructions consumed) as the RL training curve.

For each test, in order: union in its toggle hits, record the running
union size/pct AND the running instruction count consumed so far (from the
matching corpus_suite_<name>.json's "n" field), so the two curves can be
compared on equal footing.

Usage (from rl-coverage/level11_maxconfig/, after the suite has run):
    python3 dump_suite_progression_csv.py [out.csv]
"""
import sys
import glob
import json
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))

import cov_parser

TOTAL_BINS = 38696


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(THIS / "suite_progression.csv")

    cpu_dir = (THIS.parent.parent / "cpu").resolve()
    dat_paths = sorted(glob.glob(str(cpu_dir / "coverage_suite_*.dat")))
    if not dat_paths:
        print("ERROR: no coverage_suite_*.dat found in", cpu_dir)
        sys.exit(1)

    union = set()
    cum_instr = 0
    rows = []

    for i, path in enumerate(dat_paths, start=1):
        name = Path(path).stem[len("coverage_suite_"):]
        corpus_path = THIS / f"corpus_suite_{name}.json"
        n_instr = 0
        if corpus_path.exists():
            with open(corpus_path) as f:
                n_instr = json.load(f).get("n", 0)
        cum_instr += n_instr

        summary = cov_parser.parse(path)
        raw = cov_parser.hit_set(summary, cov_parser.KIND_TOGGLE)
        before = len(union)
        union |= raw
        new_bins = len(union) - before

        cum_pct = 100.0 * len(union) / TOTAL_BINS
        rows.append((i, name, n_instr, cum_instr, new_bins, len(union), cum_pct))

    with open(out_path, "w") as f:
        f.write("test_idx,test_name,n_instr,cum_instr,new_bins,cum_bins,cum_pct\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(f"Final: {rows[-1][5]} bins / {TOTAL_BINS} = {rows[-1][6]:.2f}%  "
          f"(total instr: {rows[-1][3]})")


if __name__ == "__main__":
    main()
