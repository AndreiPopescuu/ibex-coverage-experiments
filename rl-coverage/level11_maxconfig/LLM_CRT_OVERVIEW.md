# L11 LLM CRT — overview

Orientation doc for anyone joining this specific sub-project (the LLM-derived
constrained-random test generator for Ibex). Read this before diving into
individual files.

## What this is

`level11_maxconfig` runs the "max" Ibex build (PMP, ICache+scramble, RV32B,
lockstep, all on — 38,696 Verilator toggle bins) through several different
stimulus-generation strategies and compares their coverage. This particular
doc is about one of them: the **LLM CRT** (`constrained_llm_l11.py`) — a
constrained-random test generator whose constraints were derived by LLM
agents reading *only* the Ibex RTL source and this project's own Verilator
coverage output, with **zero borrowed lowRISC/riscv-dv verification
knowledge**. The research question: can that zero-domain-knowledge process
produce a test generator competitive with `constrained_random_l11.py` (a
generator hand-ported from lowRISC's actual riscv-dv logic) or the full
171-test testlist suite?

## How the CRT is built

Every stream in `constrained_llm_l11.py` went through the same three-stage
pipeline, repeated in passes as new RTL modules got targeted:

1. **Isolated synthesis** — a fresh agent, no conversation context, given
   only one RTL module's source + the specific Verilator toggle bins still
   uncovered + this project's action-space description (`codec_l11.py`'s
   6-tuple interface). Proposes `build_<name>_stream(rng)` functions, each
   docstring citing the exact RTL file:line it targets.
2. **Isolated review** — a second fresh agent, given the draft(s) + the same
   RTL, independently re-derives and checks every claim (RTL citations,
   op/csr-bucket indices, whether the codec's actual constraints — no
   label/jump-target resolution, only 5 fixed immediate buckets, no memory
   model — make the proposed stream actually work).
3. **Merge** — confirmed/fixed functions get appended to `ALL_STREAM_BUILDERS`.

The full pass-by-pass log (which modules, how many streams, what the review
caught) lives in `constrained_llm_l11.py`'s own module docstring — that's
the authoritative history, not this file.

## File map

### The CRT itself
| File | Role |
|---|---|
| `codec_l11.py` | The action space: encodes a `(op, rd, rs1, rs2, imm_bucket, csr_bucket)` 6-tuple into a real RV32IMC+RV32B instruction word. 143 ops, 5 imm buckets, ~70-entry CSR pool. Everything the CRT can possibly do is bounded by what this file can encode. |
| `constrained_llm_l11.py` | The CRT: `ALL_STREAM_BUILDERS`, currently 68 functions across 13 RTL modules. Each function's docstring is its own RTL citation/justification. |
| `testlist_l11.py` | Wires `ALL_STREAM_BUILDERS` into the `llm_rtl_directed_test` profile (`_build_llm_rtl_directed`) alongside every other profile (lowRISC-style baseline profiles, CSR sweeps, etc.) that this project's suite runner understands. |
| `llm_constraint_synth.py` | A *different*, unused-for-the-current-streams synthesis path: one big single-shot Anthropic API call instead of the isolated-agent-per-module pipeline above. Kept for reference/comparison; not what produced the committed 68 streams. |

### Corpus generation & running
| File | Role |
|---|---|
| `gen_one_profile_corpus.py` | Generates `corpus_suite_<profile>_seed*.json` for one profile, sized to hit a target instruction count. |
| `run_llm_profile.sh` | One-command run+report for `llm_rtl_directed_test`: regenerates corpus (cleans stale seed files first), runs Vtop in parallel, prints coverage. **Start here** to reproduce a number. |
| `run_1p4M.sh` | Same, at 10x the matched-suite instruction budget (to compare against lowRISC's real regression scale). |
| `gen_testlist_corpora.py` / `gen_baseline_corpus.py` | Corpus generation for other profiles / a frozen fixed-op-coverage corpus (upstream-vendor comparison, unrelated to the LLM CRT specifically). |

### Coverage analysis / reporting
| File | Role |
|---|---|
| `report_profile_coverage.py` | Union toggle/branch/line coverage across one profile's seed `.dat` files. |
| `ceiling_analysis_opentitan_upstream.py` | Theoretical max coverage: finds bins directly tied to a compile-time constant at the testbench boundary (irq lines, `debug_req_i`, scramble key/nonce, `hart_id_i`, ...). Currently: 752/38,696 dead → 98.06% ceiling. |
| `coverage_gap_report.py` | Two modes: `module <name>` (per-signal breakdown of what's still uncovered in one RTL module) and `scan` (unions every same-build `.dat` on disk, classifies remaining gaps as known-dead vs unknown, ranked by bins-explained-per-signal-traced — where to spend RTL-reading effort next). Companion to the ceiling script: catches *indirectly* dead bins (computed from constants, not directly tied) that the ceiling script's direct-tie check misses. |
| `aggregate_suite_coverage.py` / `aggregate_original171_coverage.py` | Union coverage across the full lowRISC-style testlist suite (not the LLM CRT specifically). |
| `ceiling_analysis.py` | Same idea as the `_opentitan_upstream` version, but for the older `sim_build_max` build (33,624 bins) — kept for that build's own history. |

### Comparisons against other strategies (same level, different generator)
| File | Role |
|---|---|
| `constrained_random_l11.py` | The "expert" baseline: constraints ported line-by-line from lowRISC's actual riscv-dv logic — what the LLM CRT is trying to match without having read it. |
| `compare_rl_vs_suite.py` | RL-agent-vs-testlist-suite coverage diff, by module. |
| `compare_upstream.py` | Old-vendored-RTL vs freshly-vendored-lowRISC-master toggle coverage, same frozen corpus. |
| `replay_corpus.py` / `replay_suite_corpus_dat.py` / `replay_all_suite_corpora.sh` | Re-run an already-recorded corpus against a (possibly different) build to get fresh `.dat` output. |

### RL agent (separate approach, not the LLM CRT)
`env_l11.py`, `train_l11_ppo.py`, `run_l11_steps_curriculum.sh`,
`dump_history_csv.py`, `dump_suite_progression_csv.py` — PPO-trained RL
agent targeting the same build/coverage. Independent of the LLM CRT; only
relevant here as a coverage comparison point.

### Suite runners / misc utilities
`run_scaled_upstream_suite.sh`, `run_multiseed_experiment.sh`,
`run_gen_corpus.sh`, `run_gen_constrained.sh`, `run_compare.sh`,
`watch_suite_progress.sh`, `watch_suite_progress_extended.sh` — running and
monitoring the full lowRISC-style testlist suite, not the LLM CRT
specifically. `test_rv32b.py` — one-off sanity check for the 29 RV32B ops.

### Historical reports
`RL_VS_TESTLIST_SUITE_REPORT.md`, `TESTLIST_SUITE_SUMMARY.txt`,
`UPSTREAM_COMPARISON_SUMMARY.txt` — written at the end of earlier phases of
this project; not about the LLM CRT specifically.

### Archive
`archive/llm_crt_10x_run_corpora/` — the 737 `corpus_suite_llm_rtl_directed_
test_seed*.json` files from the 10x-budget run (`run_1p4M.sh`), moved out of
the live directory so they don't shadow/contaminate a fresh matched-budget
run's corpus glob (see "open threads" below — this exact contamination
already happened once this session). Regenerate on demand with
`run_1p4M.sh` rather than relying on this snapshot; it's kept for reference,
not as something to run against directly.

## Current status / open threads

Latest clean matched-budget measurement (68 streams, 60 seeds, 139,406
instructions, `sim_build_opentitan_upstream` built 2026-07-17):
**toggle 79.54% / branch 69.42% / line 51.80%**.

There's an unresolved discrepancy against a ~83.4% figure from an earlier
LLM CRT run that was only ever seen in terminal output on a different
machine — never saved to a file or committed anywhere, so it can't be
directly re-derived. Three live, NOT yet distinguished hypotheses for the
gap:

1. **A shuffle bug** — `testlist_l11.py:284`'s `_build_llm_rtl_directed()`
   does `rng_py.shuffle(actions)` on the *entire* flattened instruction list
   from every stream combined, not just stream order. Any stream whose
   effect depends on strict instruction adjacency (compute a value, then
   immediately use it) can have that ordering destroyed. Confirmed suspect:
   the new `csr_mtvec` stream showed *zero* measured improvement despite
   being value-diversity-correct in isolation, while the near-identical
   `csr_mepc`/`csr_mtval` streams (whose CSRs also get incidental hardware-
   driven writes on any trap, independent of the shuffle) fully closed their
   gap. Not yet confirmed as the root cause, and not yet checked against how
   many of the *original* 63 streams also rely on adjacency.
2. **Seed-count dilution** — adding streams lengthens every seed's program
   (all streams run in every seed), so the same instruction budget buys
   fewer independent seeds: 74 seeds at 63 streams vs. 60 at 68. Fewer
   independent RNG draws could mean less of each stream's narrow reachable
   space gets explored, independent of whether any individual stream works.
3. **Genuine action-space ceiling** — confirmed for at least `nt_branch_addr`
   (only 5 fixed branch/jump offsets available, not enough to synthesize
   arbitrary PC bit patterns), `ibex_icache`'s fetch-address-dependent bins
   (`data_tweak_lw`/`tag_tweak_lw`/`fill_ram_req_addr`/...), and
   `ibex_compressed_decoder`'s `cm_rlist`/`cm_sp_offset`/`cm_state` (Zcmp
   push/pop ops the codec has no encoding for at all).

Ceiling refinement is also mid-flight: the automated tb-tie-off scan
(`ceiling_analysis_opentitan_upstream.py`) finds 752 directly-dead bins
(98.06% ceiling), but this session found 537 *more* dead bins one level
removed — the scramble cipher's internal key-schedule state
(`prim_prince`/`prim_subst_perm`), fed by an already-known-dead zeroed key,
is just as unreachable even though nothing ties it directly. `coverage_gap_
report.py scan` still lists ~1000 distinct signal names across the rest of
the design not yet individually traced.
