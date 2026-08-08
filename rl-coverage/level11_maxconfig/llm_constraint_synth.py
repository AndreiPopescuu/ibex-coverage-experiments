"""llm_constraint_synth.py — calls the Claude API to derive CRT constraints
for codec_l11's action space directly from Ibex RTL + this project's own
coverage data, WITHOUT referencing lowRISC's verification IP (riscv-dv/
testlist.yaml) — an independent alternative to constrained_random_l11.py's
lowRISC-ported mechanism, testing whether an LLM can derive comparable
constraints purely from reading the DUT (RTL) and this project's own
coverage feedback, instead of copying an already-solved test suite.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    # NOTE: this is an Anthropic Console API key (console.anthropic.com,
    # pay-per-token billing) — separate from a claude.ai Pro/Max chat
    # subscription, which does not by itself grant API access.

    python3 llm_constraint_synth.py                    # calls the API, writes output
    python3 llm_constraint_synth.py --dry-run           # just prints the prompt, no API call/key needed
    python3 llm_constraint_synth.py --model claude-opus-5

Writes:
    constrained_llm_l11.py            — extracted Python code (category weights
                                         + build_*_stream functions), same
                                         interface shape as
                                         constrained_random_l11.py's real
                                         functions, meant to be reviewed by
                                         hand before use — NOT auto-integrated
                                         into testlist_l11.py/env_l11.py.
    llm_constraint_synth_response.md  — full raw model response (reasoning +
                                         code), for review/audit of what RTL
                                         claims it's making.
"""
import argparse
import os
import re
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
CPU  = (THIS.parent.parent / "cpu").resolve()
L5   = (THIS.parent / "level5_real_rtl").resolve()
# Vendored RTL is spread across several fusesoc sub-packages under
# src_upstream/ (ibex_core, ibex_icache, ...) -- search recursively per
# filename rather than assuming one fixed subdir.
SRC_UPSTREAM = CPU / "src_upstream"

sys.path.insert(0, str(THIS))
sys.path.insert(0, str(L5))
from codec_l11 import L11_CSRS  # noqa: E402
import cov_parser  # noqa: E402

# Modules tracked by env_l11.py's coverage-shaping (same MODULES list) —
# reused here so the LLM sees the same module set this project already
# measures and shapes reward around.
MODULES = [
    "ibex_core", "ibex_cs_registers", "ibex_top", "ibex_if_stage",
    "ibex_top_tracing", "ibex_alu", "ibex_id_stage", "ibex_multdiv_fast",
    "ibex_ex_block", "ibex_tracer", "ibex_controller",
    "ibex_compressed_decoder", "ibex_register_file_ff",
    "ibex_load_store_unit", "ibex_decoder", "ibex_counter", "ibex_csr",
    "ibex_wb_stage", "ibex_icache", "ibex_pmp", "ibex_lockstep",
    "ibex_dummy_instr", "ibex_branch_predict",
]

# Modules with a standalone .sv file specific/small enough to be worth
# showing the LLM in full — cross-cutting glue (ibex_core/ibex_top/
# ibex_id_stage/ibex_ex_block/...) is skipped: too large and too generic
# relative to the token budget to usefully reason about here.
RTL_MODULES_OF_INTEREST = [
    "ibex_pmp", "ibex_cs_registers", "ibex_csr", "ibex_counter",
    "ibex_dummy_instr", "ibex_alu", "ibex_multdiv_fast",
    "ibex_load_store_unit", "ibex_branch_predict",
    "ibex_compressed_decoder", "ibex_decoder", "ibex_icache",
]


def _module_of(key: str) -> str | None:
    """Same extraction env_l11.py uses on cov_parser's raw point keys."""
    m = re.search(r"page\x02v_toggle/([^\x01]+)\x01", key)
    if not m:
        return None
    return m.group(1).split("__")[0]


def gather_coverage_summary(dat_paths: list[str]) -> str:
    """Aggregate the given coverage .dat file(s) into a per-module toggle-
    coverage % table, so the LLM knows what's weak.

    IMPORTANT: dat_paths defaults to EMPTY (see main()/--coverage-dat) on
    purpose. This project's cpu/coverage_suite_*.dat files on disk are
    leftovers from runs of the lowRISC-ported test suite
    (testlist_l11.py) — feeding those to the LLM here would leak lowRISC's
    own test strategy's *results* into a pass that's supposed to derive
    constraints independently of it, defeating the point. If you want a
    genuinely neutral "coverage so far" baseline instead of RTL-structure-
    only, first generate one with ZERO domain knowledge (e.g. replay
    constrained_random_l11.py's plain uniform build_actions() output, not
    any testlist_l11.py profile) and pass *that* .dat via --coverage-dat.
    """
    if not dat_paths:
        return ("(no coverage data provided — deriving from RTL structure "
                 "alone, i.e. starting from zero, see gather_coverage_summary()'s "
                 "docstring for why coverage_suite_*.dat isn't used by default)")

    dat_files = [Path(p) for p in dat_paths]
    covered = {m: set() for m in MODULES}
    totals  = {m: set() for m in MODULES}
    for f in dat_files:
        try:
            summary = cov_parser.parse(str(f))
        except Exception:
            continue
        for k, v in summary.points.items():
            key = "\x01" + k
            if "\x01page\x02v_toggle/" not in key:
                continue
            mod = _module_of(key)
            if mod not in totals:
                continue
            totals[mod].add(k)
            if v > 0:
                covered[mod].add(k)

    lines = ["module,toggle_covered,toggle_total,pct"]
    for m in MODULES:
        tot = len(totals[m])
        cov = len(covered[m])
        pct = 100.0 * cov / tot if tot else 0.0
        lines.append(f"{m},{cov},{tot},{pct:.1f}")
    return "\n".join(lines)


def _find_rtl_file(name: str) -> Path | None:
    matches = list(SRC_UPSTREAM.rglob(f"{name}.sv"))
    return matches[0] if matches else None


def gather_rtl_snippets(max_chars_per_file: int) -> str:
    parts = []
    for name in RTL_MODULES_OF_INTEREST:
        f = _find_rtl_file(name)
        if f is None:
            parts.append(f"### {name}.sv — NOT FOUND anywhere under {SRC_UPSTREAM}")
            continue
        text = f.read_text(errors="replace")
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n... [TRUNCATED — file continues] ..."
        parts.append(f"### {name}.sv\n```systemverilog\n{text}\n```")
    return "\n\n".join(parts)


def gather_action_space_summary() -> str:
    csr_lines = "\n".join(f"  csr_bucket={i}: addr=0x{addr:03x}"
                           for i, addr in enumerate(L11_CSRS))
    return f"""This project drives Ibex through a flat instruction stream encoded by a
Python codec (codec_l11.py), NOT a real assembler — every "instruction" is
chosen as a 6-tuple action (op, rd, rs1, rs2, imm_bucket, csr_bucket):

  - op: integer 0-142 selecting an instruction/opcode (RV32IMC + RV32B Zba/Zbb/
    Zbs + legacy zbp/zbc/zbe/zbf bitmanip). Categories (index ranges):
      alu 0-18,87-142 | load_store 19-26 | csr 27-29,66-68 | muldiv 30-37 |
      branch 38-43 | jump 44,65 | compressed 45-60,73-82 | upper_imm 61,64 |
      system 62,63,69-72 | exception 83-86 (illegal-instruction / misaligned
      load-store, fixed encodings, not something you can parameterize further)
  - rd, rs1, rs2: integer 0-31 (x0-x31), free choice, no assembler/register
    allocator involved — any value is always legal to emit.
  - imm_bucket: integer 0-4, mapping to ONE of exactly 5 fixed immediate/
    branch-offset values: {{0: -2048, 1: -100, 2: 0, 3: 100, 4: 2047}}. There is
    NO label/address resolution in this codec — a "backward branch" is just
    picking bucket 1 (-100 bytes), not a computed jump to a specific PC.
  - csr_bucket: integer 0-{len(L11_CSRS) - 1}, indexing into a FIXED pool of CSR
    addresses (listed below) — this is the only way any CSR read/write is
    targeted; there is no way to address a CSR not in this pool.

CSR pool (csr_bucket index -> CSR address):
{csr_lines}

Hard constraints on what you're allowed to assume: NO Spike/ISA-reference
co-simulation (no functional-correctness checking here, only RTL toggle/
branch coverage — a "correct" architectural outcome is NOT verified). NO real
memory/address-space model (loads/stores use small immediate offsets off
arbitrary base registers, not a resolved memory map — you cannot compute "the
address of PMP region N's boundary"). NO sub-program/label/jump-target
mechanism. NO privilege-mode-switch prologue (boot mode is fixed for a given
run). If an RTL structure needs one of these to exercise (e.g. an address
comparator needing a specific resolved address), say so explicitly rather
than inventing a stream that can't actually work in this codec.
"""


PROMPT_TEMPLATE = """You are deriving constrained-random test-generation rules for an \
Ibex RISC-V CPU core, targeting RTL toggle/branch coverage in a Verilator \
testbench. You have been given:

1. The actual SystemVerilog RTL source for the modules this project measures \
coverage on (below).
2. Per-module toggle-coverage percentages, IF any coverage data was provided \
(below) — lower % means that module needs more/different stimulus. If this \
section says no data was provided, you are deriving constraints from a clean \
slate (zero prior coverage) — reason from RTL structure alone in that case, \
don't invent coverage numbers.
3. A description of the ONLY action space available to generate stimulus (a \
flat 6-tuple instruction encoding, below) — any constraint you propose MUST \
be expressible as Python code choosing values for this exact 6-tuple, nothing \
else (no real assembler, no label resolution, no memory model — see the hard \
constraints at the end of that section).

IMPORTANT: Do NOT reference, assume, or rely on lowRISC's own verification IP \
(riscv-dv, UVM testbenches, testlist.yaml, or any of their existing test \
names/strategies) — derive constraints purely from reading the RTL structure \
and the coverage numbers given to you below. The goal is an INDEPENDENT \
derivation, not a reproduction of lowRISC's own test suite.

Your job: propose Python code, in EXACTLY this interface shape:

  CATEGORY_WEIGHTS_LLM: dict[str, float] — category -> probability, summing to
      1.0, categories are exactly: alu, load_store, csr, muldiv, branch, jump,
      compressed, upper_imm, system, exception (see action space description).
      Explain (in a comment) WHY you chose non-uniform weights, if you do,
      tied to specific RTL structures/coverage gaps below — don't just guess
      a plausible-sounding distribution.

  One or more build_<name>_stream(rng) -> list[tuple[int, int, int, int, int,
      int]] functions (each element is (op, rd, rs1, rs2, imm_bucket,
      csr_bucket)), each targeting a SPECIFIC uncovered/weak RTL structure you
      identify below (e.g. a specific always_comb branch, a specific case arm,
      a specific comparator, a specific FSM state transition). In each
      function's docstring, cite the EXACT RTL file + signal/case-arm/state
      you're targeting and WHY the action sequence you build would toggle/
      exercise it. If you cannot point to a specific RTL structure a stream
      would target, do not invent one — flat CATEGORY_WEIGHTS_LLM already
      covers the general/unfocused case.

Output ONLY a single ```python fenced code block containing this, plus a short
prose section (outside the code block) explaining your reasoning per stream,
referencing the specific RTL lines/signals that justify it.

── Per-module toggle coverage (empty = starting from zero) ──────────────────
{coverage_summary}

── Action space (the ONLY thing your code may produce) ─────────────────────
{action_space}

── RTL source for the tracked modules ───────────────────────────────────────
{rtl_snippets}
"""


def build_prompt(max_chars_per_file: int, coverage_dat: list[str]) -> str:
    return PROMPT_TEMPLATE.format(
        coverage_summary=gather_coverage_summary(coverage_dat),
        action_space=gather_action_space_summary(),
        rtl_snippets=gather_rtl_snippets(max_chars_per_file),
    )


def call_claude(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_code_block(text: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-sonnet-5",
                     help="API model id (default: claude-sonnet-5). Use "
                          "claude-opus-5 for a (slower, pricier) stronger-"
                          "reasoning pass if the sonnet output looks shallow.")
    ap.add_argument("--max-chars-per-rtl-file", type=int, default=12000,
                     help="Truncation limit per .sv file sent (default 12000 "
                          "chars ~= 3000 tokens; raise for deeper RTL context "
                          "at proportionally higher API cost).")
    ap.add_argument("--out-code", default=str(THIS / "constrained_llm_l11.py"))
    ap.add_argument("--out-response", default=str(THIS / "llm_constraint_synth_response.md"))
    ap.add_argument("--coverage-dat", nargs="*", default=[],
                     help="Path(s) to coverage .dat file(s) to summarize as "
                          "'coverage so far' context. Empty by default ON "
                          "PURPOSE, i.e. starting from zero — do NOT point this "
                          "at cpu/coverage_suite_*.dat, those are leftovers from "
                          "the lowRISC-ported test suite and would leak its "
                          "strategy's results into a pass meant to be "
                          "independent of it. Only pass .dat file(s) from a "
                          "run that used zero domain knowledge (e.g. "
                          "constrained_random_l11.py's plain uniform "
                          "build_actions(), not any testlist_l11.py profile).")
    ap.add_argument("--dry-run", action="store_true",
                     help="Build and print the prompt without calling the API "
                          "(no ANTHROPIC_API_KEY needed) — use this first to "
                          "sanity-check what context would be sent and its "
                          "approximate size/cost.")
    args = ap.parse_args()

    prompt = build_prompt(args.max_chars_per_rtl_file, args.coverage_dat)

    if args.dry_run:
        print(prompt)
        print(f"\n[dry-run] prompt length: {len(prompt)} chars "
              f"(~{len(prompt) // 4} tokens, rough estimate)", file=sys.stderr)
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set.\n\n"
            "This must be an Anthropic Console API key (console.anthropic.com,\n"
            "pay-per-token billing) — separate from a claude.ai Pro/Max chat\n"
            "subscription, which does not by itself grant API access.\n\n"
            "Create a key there, then:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "and re-run this script. Use --dry-run first if you just want to\n"
            "see what would be sent, without a key."
        )

    print(f"[llm_constraint_synth] calling {args.model} "
          f"(prompt ~{len(prompt) // 4} tokens)...", file=sys.stderr)
    response_text = call_claude(prompt, args.model)

    Path(args.out_response).write_text(response_text)
    print(f"[llm_constraint_synth] wrote raw response -> {args.out_response}")

    code = extract_code_block(response_text)
    if not code:
        sys.exit("No ```python code block found in the response — see "
                  f"{args.out_response} to inspect what came back instead.")

    header = (
        '"""constrained_llm_l11.py — CRT constraints derived by an LLM reading\n'
        "Ibex RTL + this project's own coverage data directly (NOT ported from\n"
        "lowRISC's riscv-dv/testlist.yaml — see constrained_random_l11.py for\n"
        "that version). Generated by llm_constraint_synth.py; UNREVIEWED — read\n"
        "before trusting any RTL claim it makes, the same way you'd review any\n"
        'other unverified source before acting on it.\n"""\n\n'
    )
    Path(args.out_code).write_text(header + code)
    print(f"[llm_constraint_synth] wrote extracted constraints -> {args.out_code}")
    print("[llm_constraint_synth] NOT auto-integrated into testlist_l11.py/"
          "env_l11.py — review constrained_llm_l11.py by hand first, the code "
          "in it has not been syntax-checked or run.")


if __name__ == "__main__":
    main()
