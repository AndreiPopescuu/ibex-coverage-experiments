"""compare_upstream.py — compare toggle/branch/line coverage between the
project's old-vendored Ibex RTL (cpu/sim_build_max/, coverage_max_baseline.dat)
and a freshly-vendored lowRISC/ibex master build (cpu/sim_build_max_upstream/,
coverage_max_upstream.dat), driven by the SAME frozen stimulus corpus
(corpus_max_baseline_v1.json) so the only variable is RTL vintage.

Also reports a cheap proxy for the "27 legacy-draft RV32B ops (zbp/zbc/zbe/zbf)
may now decode as illegal instructions" finding: count how many times each
run's ibex_tracer trace log shows the trap handler (0x00200000) being entered.
A large jump on the upstream build is consistent with those ops trapping,
since ratified RV32B (which upstream ibex_decoder.sv implements) dropped them.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPU = os.path.join(_HERE, "..", "..", "cpu")
sys.path.insert(0, os.path.join(_HERE, "..", "level5_real_rtl"))

import cov_parser  # noqa: E402

OLD_DAT = os.path.join(_CPU, "coverage_max_baseline.dat")
NEW_DAT = os.path.join(_CPU, "coverage_max_upstream.dat")
# trace_core_00000000.log is overwritten by every run (written to cwd
# regardless of which Vtop produced it) — both run scripts snapshot it to a
# distinct name immediately after running, see cpu/run_old_baseline.sh and
# cpu/run_upstream_corpus.sh.
OLD_TRACE = os.path.join(_CPU, "trace_core_00000000_old_baseline.log")
NEW_TRACE = os.path.join(_CPU, "trace_core_00000000_upstream.log")
PROVENANCE = os.path.join(_CPU, "src_upstream", "_UPSTREAM_PROVENANCE.txt")


def _trap_count(trace_path):
    if not os.path.exists(trace_path):
        return None
    n = 0
    with open(trace_path, errors="ignore") as f:
        for line in f:
            cols = line.split("\t")
            if len(cols) > 2 and cols[2].strip() == "00200000":
                n += 1
    return n


def main():
    if not os.path.exists(OLD_DAT):
        print(f"ERROR: missing {OLD_DAT} — run cpu/run_old_baseline.sh first")
        sys.exit(1)
    if not os.path.exists(NEW_DAT):
        print(f"ERROR: missing {NEW_DAT} — build+run Makefile.upstream first")
        sys.exit(1)

    old = cov_parser.parse(OLD_DAT)
    new = cov_parser.parse(NEW_DAT)

    if os.path.exists(PROVENANCE):
        print("Upstream RTL provenance:")
        print("  " + open(PROVENANCE).read().replace("\n", "\n  ").strip())
        print()

    print("=" * 78)
    print(f"{'kind':8s} {'OLD cov/tot':>16s} {'OLD %':>8s}   {'NEW cov/tot':>16s} {'NEW %':>8s}   {'delta':>8s}")
    print("=" * 78)
    for kind in (cov_parser.KIND_TOGGLE, cov_parser.KIND_BRANCH, cov_parser.KIND_LINE):
        oc, ot = old.by_kind.get(kind, (0, 0))
        nc, nt = new.by_kind.get(kind, (0, 0))
        op = old.kind_pct(kind)
        npct = new.kind_pct(kind)
        print(f"{kind:8s} {oc:>7}/{ot:<8} {op:7.2f}%   {nc:>7}/{nt:<8} {npct:7.2f}%   {npct - op:+7.2f}%")
    print()

    old_pages = set(old.by_page)
    new_pages = set(new.by_page)
    only_new = sorted(new_pages - old_pages)
    only_old = sorted(old_pages - new_pages)
    common = sorted(old_pages & new_pages)

    print(f"Modules (pages) only in NEW (upstream) build: {len(only_new)}")
    for p in only_new[:20]:
        c, t = new.by_page[p]
        print(f"  + {p}  ({c}/{t} = {100*c/t if t else 0:.1f}%)")
    if len(only_new) > 20:
        print(f"  ... and {len(only_new) - 20} more")
    print()

    print(f"Modules (pages) only in OLD (vendored) build: {len(only_old)}")
    for p in only_old[:20]:
        c, t = old.by_page[p]
        print(f"  - {p}  ({c}/{t} = {100*c/t if t else 0:.1f}%)")
    if len(only_old) > 20:
        print(f"  ... and {len(only_old) - 20} more")
    print()

    print("Biggest coverage % deltas among modules present in BOTH builds:")
    deltas = []
    for p in common:
        oc, ot = old.by_page[p]
        nc, nt = new.by_page[p]
        opct = 100 * oc / ot if ot else 0
        npct = 100 * nc / nt if nt else 0
        deltas.append((npct - opct, p, opct, npct))
    deltas.sort(key=lambda x: abs(x[0]), reverse=True)
    for d, p, opct, npct in deltas[:15]:
        print(f"  {d:+6.1f}%  {p:45s} old={opct:5.1f}%  new={npct:5.1f}%")
    print()

    print("-" * 78)
    print("Legacy-draft RV32B check (zbp/zbc/zbe/zbf, 27/143 corpus ops):")
    old_traps = _trap_count(OLD_TRACE)
    new_traps = _trap_count(NEW_TRACE)
    print(f"  trap-handler (PC=0x00200000) entries — OLD: {old_traps}   NEW: {new_traps}")
    if old_traps is not None and new_traps is not None:
        if new_traps > old_traps:
            print(f"  -> NEW build trapped {new_traps - old_traps} more times than OLD on the same "
                  f"corpus. Consistent with ratified RV32B (upstream) rejecting the legacy-draft "
                  f"zbp/zbc/zbe/zbf encodings that the old vendored decoder still accepted as legal.")
        else:
            print(f"  -> No trap-count increase observed; legacy-draft ops may still decode "
                  f"identically, or this proxy didn't catch the difference (traps can also come "
                  f"from other corpus instructions, e.g. PMP/CSR edge cases) — not conclusive on "
                  f"its own without an isolated per-op corpus.")
    print("-" * 78)


if __name__ == "__main__":
    main()
