"""coverage_gap_report.py — two diagnostics for figuring out WHY a toggle-bin
is still uncovered, built while investigating why the LLM CRT plateaus well
below ceiling_analysis_opentitan_upstream.py's 98.06% theoretical max.

Companion to that script: ceiling_analysis_opentitan_upstream.py only catches
bins DIRECTLY tied to a compile-time constant at the testbench boundary. It
misses bins that are only INDIRECTLY dead -- computed purely from other
constant signals, without being a literal port tie (e.g. this project's
scramble-cipher internals: fed a zeroed key_i, their own key-derived state
is just as unreachable as the key port itself, but no script catches that
without tracing the RTL fan-in one signal at a time). This script narrows
down WHERE to spend that RTL-tracing effort.

Two modes:

  module   <module_name> [--profile NAME]
      Per-signal breakdown of what's still uncovered in one RTL module, for
      one profile's coverage_suite_<profile>_seed*.dat union. Use this to
      see exactly which base signals (grouped, bit-index stripped) still
      have missing toggle bins, ranked by how many bits are missing --
      the same view used to scope every Pass 3 synthesis-agent prompt.

  scan     [--focus mod1,mod2,...]
      Unions EVERY *.dat file on disk that matches the reference build's
      bin count (skips any from a different build -- see REFERENCE_TOTAL --
      mixing builds silently produces nonsense, hit-count > total), splits
      the still-missing bins into already-known-dead (imported from
      ceiling_analysis_opentitan_upstream.DEAD_SIGNALS, not duplicated here)
      vs unknown-status, then reports which modules have the best
      bins-explained-per-distinct-signal-traced ratio -- i.e. where a
      handful of RTL reads could resolve the most bins, cheaply. Pass
      --focus to print the exact signal list for specific modules (what you
      hand to an RTL-tracing pass next).

Usage:
    cd rl-coverage/level11_maxconfig
    python3 coverage_gap_report.py module ibex_icache --profile llm_rtl_directed_test
    python3 coverage_gap_report.py scan
    python3 coverage_gap_report.py scan --focus prim_subst_perm,ibex_counter,prim_prince,ibex_csr
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPU = os.path.join(_HERE, "..", "..", "cpu")
sys.path.insert(0, os.path.join(_HERE, "..", "level5_real_rtl"))
sys.path.insert(0, _HERE)
import cov_parser  # noqa: E402
from ceiling_analysis_opentitan_upstream import DEAD_SIGNALS  # noqa: E402

_PAGE_MOD_RE = re.compile(r"page\x02v_toggle/([^\x01]+)\x01")
_O_RE = re.compile(r"\x01o\x02([^\x01]+)")
_BIT_SUFFIX_RE = re.compile(r"\[\d+\]$")

# The opentitan_upstream build's total toggle-bin count. Any .dat whose own
# total differs is from a DIFFERENT build (sim_build_max, sim_build_max_
# upstream, ...) and must be excluded from a union -- their bin namespaces
# aren't compatible, and mixing them silently produces hit > total.
REFERENCE_TOTAL = 38696


def module_of(key: str) -> str | None:
    m = _PAGE_MOD_RE.search(key)
    return m.group(1).split("__")[0] if m else None


def signal_of(key: str) -> str | None:
    m = _O_RE.search(key)
    return m.group(1) if m else None


def base_signal(sig: str) -> str:
    return _BIT_SUFFIX_RE.sub("", sig)


def cmd_module(args):
    paths = sorted(glob.glob(os.path.join(_CPU, f"coverage_suite_{args.profile}_seed*.dat")))
    if not paths:
        sys.exit(f"no coverage_suite_{args.profile}_seed*.dat found in {_CPU} -- run it first")
    print(f"{len(paths)} seed dat files")

    hit = set()
    for p in paths:
        s = cov_parser.parse(p)
        hit |= cov_parser.hit_set(s, cov_parser.KIND_TOGGLE)

    s0 = cov_parser.parse(paths[0])
    all_keys = [k for k in s0.points if "page\x02v_toggle/" in ("\x01" + k)]
    mod_keys = [k for k in all_keys if module_of(k) == args.module]
    if not mod_keys:
        sys.exit(f"module {args.module!r} not found in this build's toggle bins")
    mod_hit = [k for k in mod_keys if k in hit]
    mod_miss = [k for k in mod_keys if k not in hit]

    print(f"{args.module}: {len(mod_hit)}/{len(mod_keys)} toggle bins hit "
          f"by {args.profile}'s union ({100.0 * len(mod_hit) / len(mod_keys):.2f}%)")

    sig_counter = Counter()
    for k in mod_miss:
        sig = signal_of(k)
        if sig:
            sig_counter[base_signal(sig)] += 1

    print(f"\nUncovered signals in {args.module} (base name, bit count missing), "
          f"'*DEAD*' = already known unreachable:")
    for sig, n in sig_counter.most_common(60):
        tag = " *DEAD*" if sig in DEAD_SIGNALS else ""
        print(f"  {sig:<40} {n}{tag}")


def cmd_scan(args):
    paths = sorted(set(glob.glob(os.path.join(_CPU, "*.dat")) +
                        glob.glob(os.path.join(_CPU, "coverage_suite_*_seed*.dat"))))
    print(f"{len(paths)} .dat files found, filtering to {REFERENCE_TOTAL}-bin build...")

    hit = set()
    all_keys = None
    n_used = 0
    for p in paths:
        try:
            s = cov_parser.parse(p)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: parse error {e!r}")
            continue
        keys = [k for k in s.points if "page\x02v_toggle/" in ("\x01" + k)]
        if len(keys) != REFERENCE_TOTAL:
            continue  # different build, silently skip (too noisy to log every one)
        if all_keys is None:
            all_keys = keys
        hit |= cov_parser.hit_set(s, cov_parser.KIND_TOGGLE)
        n_used += 1
    print(f"{n_used} files actually used (same-build)")

    if all_keys is None:
        sys.exit(f"no .dat file with {REFERENCE_TOTAL} bins found -- nothing to scan")

    total = len(all_keys)
    miss = [k for k in all_keys if k not in hit]
    print(f"total toggle bins: {total}, hit (union of everything on disk): {len(hit)}, "
          f"still missing: {len(miss)}")

    known_dead, unknown = [], []
    for k in miss:
        sig = signal_of(k)
        base = base_signal(sig) if sig else None
        (known_dead if base in DEAD_SIGNALS else unknown).append(k)

    print(f"of the {len(miss)} missing bins: {len(known_dead)} already known-dead (tied off), "
          f"{len(unknown)} UNKNOWN status -- these need RTL tracing")

    by_mod = defaultdict(Counter)
    for k in unknown:
        by_mod[module_of(k) or "?"][base_signal(signal_of(k) or "?")] += 1

    print(f"\n{len(by_mod)} modules have unknown-status bins, ranked by "
          f"bins-per-distinct-signal (cheapest to trace first):")
    ranked = sorted(by_mod.items(),
                     key=lambda kv: -(sum(kv[1].values()) / max(1, len(kv[1]))))
    for mod, ctr in ranked:
        bins, n_sig = sum(ctr.values()), len(ctr)
        print(f"  {mod:<26} {bins:>5} bins, {n_sig:>3} distinct signals "
              f"({bins / n_sig:.0f} bins/signal)")

    if args.focus:
        focus_mods = [m.strip() for m in args.focus.split(",")]
        print(f"\n=== Focus modules: {focus_mods} ===")
        for mod in focus_mods:
            ctr = by_mod.get(mod, Counter())
            print(f"\n{mod}: {sum(ctr.values())} bins, {len(ctr)} distinct signals")
            for sig, n in ctr.most_common():
                print(f"  {sig:<40} {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_mod = sub.add_parser("module", help="per-signal breakdown for one RTL module")
    p_mod.add_argument("module", help="e.g. ibex_icache")
    p_mod.add_argument("--profile", default="llm_rtl_directed_test")
    p_mod.set_defaults(func=cmd_module)

    p_scan = sub.add_parser("scan", help="whole-design known-dead vs unknown scan")
    p_scan.add_argument("--focus", default=None,
                         help="comma-separated module names to print full signal lists for")
    p_scan.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
