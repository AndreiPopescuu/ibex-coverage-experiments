"""replay_corpus.py — replay an existing corpus (recorded against the minimal
Ibex build) on the "max" build, and report the resulting coverage baseline.

Build the matching Vtop once with:
    cd cpu && make CONFIG=max

This is step 1 of the minimal->max workflow: programs that were good enough
to reach 71.94% on the minimal build are valid RV32IMC and should still run
on the max build, now exercising modules that were tied off before (icache,
pmp, wb_stage) plus their lockstep duplicates — for free, without writing
any new stimulus.

Usage:
    python replay_corpus.py --corpus ../level9_ops/corpus_all.json
"""

import argparse
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import sysconfig
from pathlib import Path

THIS   = Path(__file__).resolve().parent
ML4DV  = (THIS.parent.parent / "cpu").resolve()
VTOP   = ML4DV / "sim_build_max" / "Vtop"
COVDAT = ML4DV / "coverage_max.dat"
PROGRAM_JSON = "/tmp/rl_l11_replay.json"

sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
import cov_parser

_F_RE     = re.compile(r"\x01f\x02[^\x01]+")
_N_RE     = re.compile(r"\x01n\x02[^\x01]+")
_H_DOT_RE = re.compile(r"(\x01h\x02)\.")

def norm(key):
    k = _F_RE.sub("", key)
    k = _N_RE.sub("", k)
    k = _H_DOT_RE.sub(r"\1", k)
    return k


def run_words(words: list[int]):
    with open(PROGRAM_JSON, "w") as f:
        json.dump({"n": len(words), "agent": "l11_replay",
                   "machine_code": [int(w) for w in words]}, f)

    env = os.environ.copy()
    site_packages = sysconfig.get_paths()["purelib"]
    cocotb_libs = os.path.join(site_packages, "cocotb", "libs")
    env["LD_LIBRARY_PATH"] = (
        cocotb_libs + ":/usr/lib/x86_64-linux-gnu"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = (
        str(ML4DV) + ":" + site_packages
        + ":" + env.get("PYTHONPATH", "")
    )
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = PROGRAM_JSON

    try:
        proc = subprocess.run(
            [str(VTOP), f"+verilator+coverage+file+{COVDAT}"],
            cwd=str(ML4DV), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(COVDAT))


_stop_requested = False

def _sigint_handler(sig, frame):
    global _stop_requested
    print("\n[!] Stop cerut — salvez ce am acumulat până acum...", flush=True)
    _stop_requested = True

signal.signal(signal.SIGINT, _sigint_handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="Corpus JSON recorded on the minimal build "
                         "(e.g. ../level9_ops/corpus_all.json)")
    ap.add_argument("--save-hits", default="l11_baseline_hits.pkl",
                    help="Where to save the cumulative hit set (pickle), "
                         "for use as --hits in train_l11_ppo.py")
    ap.add_argument("--max-programs", type=int, default=None,
                    help="Rulează doar primele N programe din corpus, apoi "
                         "se oprește (util pentru un baseline rapid)")
    ap.add_argument("--start", type=int, default=0,
                    help="Sări peste primele N programe (util pentru resume "
                         "după un crash/timeout)")
    args = ap.parse_args()

    if not VTOP.exists():
        sys.exit(f"Max-config Vtop not found at {VTOP}. "
                 f"Run `cd cpu && make CONFIG=max` first.")

    with open(args.corpus) as f:
        corpus = json.load(f)

    programs  = corpus["programs"]
    total_prg = len(programs)

    s = run_words([0x00000013] * 8)
    TOTAL = s.by_kind["toggle"][1] if s else 0
    print(f"Total bins toggle (max config): {TOTAL}")
    print(f"Corpus: {total_prg} programe | coverage pe minimal: {corpus['final_cum_pct']}%\n")

    cum_hits = set()
    failed   = 0

    save_path = Path(args.save_hits)
    if save_path.exists():
        with open(save_path, "rb") as f:
            cum_hits = pickle.load(f)
        cum_pct = 100.0 * len(cum_hits) / TOTAL if TOTAL else 0.0
        print(f"[i] Resume din {save_path}: {len(cum_hits):,} hits ({cum_pct:.2f}%)\n")

    def _save():
        with open(save_path, "wb") as f:
            pickle.dump(cum_hits, f)

    try:
        for i, prog in enumerate(programs):
            if i < args.start:
                continue
            if args.max_programs and (i - args.start) >= args.max_programs:
                print(f"\n[!] --max-programs {args.max_programs} atins, mă opresc.")
                break

            summary = run_words(prog["words"])

            if summary is None:
                print(f"  [{i+1:>3}/{total_prg}] ep={prog['ep']:>4}: Vtop FAILED")
                failed += 1
                continue

            ep_hits  = {norm(k) for k, v in summary.points.items()
                        if v > 0 and "\x01page\x02v_toggle/" in ("\x01" + k)}
            new_hits = ep_hits - cum_hits
            cum_hits |= ep_hits

            cum_pct = 100.0 * len(cum_hits) / TOTAL if TOTAL else 0.0
            print(f"  [{i+1:>3}/{total_prg}] ep={prog['ep']:>4}: "
                  f"+{len(new_hits):>4} bins  |  cum {cum_pct:.2f}%")

            _save()

            if _stop_requested:
                break
    finally:
        final_pct = 100.0 * len(cum_hits) / TOTAL if TOTAL else 0.0
        print(f"\n{'='*50}")
        print(f"Coverage replay pe config maxima: {final_pct:.2f}%  ({len(cum_hits):,} / {TOTAL:,} bins)")
        print(f"Coverage acelasi corpus pe minimal: {corpus['final_cum_pct']}%")
        print(f"Programe esuate: {failed} / {total_prg}")

        _save()
        print(f"\nSaved {len(cum_hits):,} hits -> {save_path} "
              f"(use as --hits in train_l11_ppo.py)")


if __name__ == "__main__":
    main()
