# L11 Coverage History — opentitan upstream config

Verilator toggle/branch/line coverage on the `opentitan` ibex preset
(`SecureIbex=1, PMPEnable=1, ICache+ECC+Scramble, RV32B OTEarlGrey, lockstep`).

Profile: `llm_rtl_directed_test` (46 seeds unless noted).
Ceiling: **100%** structural (0 dead bins on 93768-bin build).

---

## Old build — 38,696 toggle bins (IRQ tied to 0, no lockstep coverage)

| Date | Event | Toggle | Branch | Line | Seeds/Streams |
|------|-------|--------|--------|------|---------------|
| ~Sep 01 | Baseline constrained-random | 79.54% | 69.42% | 51.80% | — |
| Sep 08 | Pass 1 (9 modules, 40 streams) | ~82% | — | — | 40 streams |
| Sep 10 | Pass 2 (+ibex_core/decoder/icache, 64 streams) | ~83% | — | — | 64 streams |
| Sep 11 | Pass 3 (exception CSRs + bt_a_mux_sel_o, 68 streams) | 84.00% | — | — | 68 streams |
| Sep 12 | Pass 4 (+ibex_counter/csr/alu/decoder, 79 streams) | **84.03%** (32518/38696) | 70.65% | 54.40% | 79 streams |

---

## New build — 93,768 toggle bins (IRQ live, lockstep shadow elaborated)

Build change: IRQ ports exposed → lockstep + full interrupt tree elaborated.
+55k bins vs old build. REFERENCE_TOTAL = 93768.

| Date | Event | Toggle | Branch | Line | Notes |
|------|-------|--------|--------|------|-------|
| Sep 13 | First run, IRQ ports exposed (VPI driver, broken) | 77.75% (72872/93726) | 72.14% | 54.30% | VPI silent no-op |
| Sep 19 | LFSR IRQ generator (commit 00e23ab) | **80.14%** (75141/93768) | 73.60% | 54.24% | +2.39pp, IRQ paths covered |
| — | Pass 5 (in progress) | — | — | — | PMP sequences + … |

---

## Gap summary (Sep 19, after LFSR fix)

18,627 bins missing, top modules:

| Module | Missing bins | Bins/signal | Priority | Notes |
|--------|-------------|-------------|----------|-------|
| ibex_cs_registers | 4,895 | 1 | HIGH | PMP cfg 1→0, mseccfg, IRQ cause fields |
| ibex_top | 1,392 | 1 | MED | top-level routing |
| ibex_core | 2,443 | 2 | MED | PMP req, crash_dump, RVFI bits |
| prim_prince | 1,157 | 1 | LOW? | ICache scramble cipher (key valid=0) |
| ibex_alu | 1,109 | 1 | MED | ALU extended ops |
| ibex_icache | 908 | 1 | MED | ICache paths |
| ibex_top_tracing | 936 | 1 | LOW | tracer signals |
| ibex_counter | 836 | 2 | MED | counter overflow paths |
| prim_ram_1p_scr | 1,100 | 2 | LOW | scramble RAM |
| ibex_tracer | 347 | 1 | LOW | tracer only |
| ibex_if_stage | 449 | 1 | MED | fetch paths |

---

## How to update this file

After each `run_llm_profile.sh` run, add a row to the table with:
- Date, event description, toggle/branch/line % from `report_profile_coverage.py`
- Number of streams in `constrained_llm_l11.py` at time of run
