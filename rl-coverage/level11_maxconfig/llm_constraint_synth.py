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

    python3 llm_constraint_synth.py                    # single call, all modules
    python3 llm_constraint_synth.py --per-module        # one call per RTL module (parallel)
    python3 llm_constraint_synth.py --dry-run           # print prompts, no API call
    python3 llm_constraint_synth.py --model claude-opus-5

--per-module mode: makes N+1 parallel API calls (one per RTL module + one
dedicated call for CATEGORY_WEIGHTS). Each module gets its full .sv file
without truncation since it's the only RTL in its prompt. Results are merged
into a single constrained_llm_l11.py. Raw responses saved as
llm_synth_response_<module>.md and llm_synth_response_weights.md.

Writes:
    constrained_llm_l11.py            — extracted Python code (category weights
                                         + build_*_stream functions), same
                                         interface shape as
                                         constrained_random_l11.py's real
                                         functions, meant to be reviewed by
                                         hand before use — NOT auto-integrated
                                         into testlist_l11.py/env_l11.py.
    llm_constraint_synth_response.md  — full raw model response in single mode.
    llm_synth_response_<name>.md      — per-module raw responses (--per-module).
    llm_synth_response_weights.md     — weights call raw response (--per-module).
"""
import argparse
import concurrent.futures
import os
import re
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
CPU  = (THIS.parent.parent / "cpu").resolve()
L5   = (THIS.parent / "level5_real_rtl").resolve()
SRC_UPSTREAM = CPU / "src_upstream"

sys.path.insert(0, str(THIS))
sys.path.insert(0, str(L5))
from codec_l11 import L11_CSRS  # noqa: E402
import cov_parser  # noqa: E402

MODULES = [
    "ibex_core", "ibex_cs_registers", "ibex_top", "ibex_if_stage",
    "ibex_top_tracing", "ibex_alu", "ibex_id_stage", "ibex_multdiv_fast",
    "ibex_ex_block", "ibex_tracer", "ibex_controller",
    "ibex_compressed_decoder", "ibex_register_file_ff",
    "ibex_load_store_unit", "ibex_decoder", "ibex_counter", "ibex_csr",
    "ibex_wb_stage", "ibex_icache", "ibex_pmp", "ibex_lockstep",
    "ibex_dummy_instr", "ibex_branch_predict",
]

RTL_MODULES_OF_INTEREST = [
    "ibex_pmp", "ibex_cs_registers", "ibex_csr", "ibex_counter",
    "ibex_dummy_instr", "ibex_alu", "ibex_multdiv_fast",
    "ibex_load_store_unit", "ibex_branch_predict",
    "ibex_compressed_decoder", "ibex_decoder", "ibex_icache",
]


def _module_of(key: str) -> str | None:
    m = re.search(r"page\x02v_toggle/([^\x01]+)\x01", key)
    if not m:
        return None
    return m.group(1).split("__")[0]


def gather_coverage_summary(dat_paths: list[str]) -> str:
    """Aggregate coverage .dat file(s) into a per-module toggle % table.

    IMPORTANT: dat_paths defaults to EMPTY on purpose — cpu/coverage_suite_*.dat
    files are leftovers from the lowRISC-ported test suite and would leak its
    strategy's results into a pass meant to be independent of it.
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


def gather_rtl_for_module(name: str, max_chars: int) -> str:
    f = _find_rtl_file(name)
    if f is None:
        return f"### {name}.sv — NOT FOUND anywhere under {SRC_UPSTREAM}"
    text = f.read_text(errors="replace")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n... [TRUNCATED — file continues] ..."
    return f"### {name}.sv\n```systemverilog\n{text}\n```"


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


# ---------------------------------------------------------------------------
# Single-call mode (original behaviour)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-module mode prompts
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_MODULE = """You are deriving constrained-random test-generation rules for \
one specific module of the Ibex RISC-V CPU core, targeting RTL toggle/branch \
coverage in a Verilator testbench.

You are working on: {module_name}

IMPORTANT: Do NOT include CATEGORY_WEIGHTS_LLM — that is handled by a separate \
call. Focus ONLY on build_*_stream functions for THIS module.

Do NOT reference lowRISC's own verification IP (riscv-dv, UVM testbenches, \
testlist.yaml) — derive constraints purely from the RTL source below.

Your job: propose ONE or more build_{module_name}_<suffix>_stream(rng) functions \
-> list[tuple[int, int, int, int, int, int]], where each element is \
(op, rd, rs1, rs2, imm_bucket, csr_bucket). Each function must target a SPECIFIC \
RTL structure in {module_name}.sv (a specific always_comb branch, case arm, \
comparator, FSM state transition). In each function's docstring, cite the EXACT \
signal/case-arm/state you're targeting and WHY the sequence would toggle it.

If you cannot point to a specific RTL structure, do not invent a stream.

Output ONLY a single ```python fenced code block containing the function(s), \
plus a short prose section explaining your reasoning, referencing specific \
RTL lines/signals.

── Per-module toggle coverage (empty = starting from zero) ──────────────────
{coverage_summary}

── Action space (the ONLY thing your code may produce) ─────────────────────
{action_space}

── RTL source for {module_name} ─────────────────────────────────────────────
{rtl_snippet}
"""

PROMPT_TEMPLATE_WEIGHTS = """You are deriving CATEGORY_WEIGHTS_LLM for constrained-random \
test generation targeting Ibex RISC-V CPU RTL toggle/branch coverage.

The following RTL modules are tracked. Based on their names and typical RTL \
complexity, propose a CATEGORY_WEIGHTS_LLM dict that prioritises instruction \
categories most likely to exercise these modules.

Modules tracked:
{module_list}

Per-module toggle coverage:
{coverage_summary}

Action space categories (from the codec):
  alu 0-18,87-142 | load_store 19-26 | csr 27-29,66-68 | muldiv 30-37 |
  branch 38-43 | jump 44,65 | compressed 45-60,73-82 | upper_imm 61,64 |
  system 62,63,69-72 | exception 83-86

Output ONLY a single ```python fenced code block containing:
  CATEGORY_WEIGHTS_LLM: dict[str, float]  # must sum to 1.0
with comments explaining WHY each weight was chosen tied to specific modules above.
"""


def build_prompt(max_chars_per_file: int, coverage_dat: list[str]) -> str:
    return PROMPT_TEMPLATE.format(
        coverage_summary=gather_coverage_summary(coverage_dat),
        action_space=gather_action_space_summary(),
        rtl_snippets=gather_rtl_snippets(max_chars_per_file),
    )


def build_module_prompt(name: str, max_chars: int, coverage_dat: list[str]) -> str:
    return PROMPT_TEMPLATE_MODULE.format(
        module_name=name,
        coverage_summary=gather_coverage_summary(coverage_dat),
        action_space=gather_action_space_summary(),
        rtl_snippet=gather_rtl_for_module(name, max_chars),
    )


def build_weights_prompt(coverage_dat: list[str]) -> str:
    return PROMPT_TEMPLATE_WEIGHTS.format(
        module_list="\n".join(f"  - {m}" for m in RTL_MODULES_OF_INTEREST),
        coverage_summary=gather_coverage_summary(coverage_dat),
    )


def call_claude(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_code_block(text: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def run_single(args) -> None:
    prompt = build_prompt(args.max_chars_per_rtl_file, args.coverage_dat)

    if args.dry_run:
        print(prompt)
        print(f"\n[dry-run] prompt length: {len(prompt)} chars "
              f"(~{len(prompt) // 4} tokens, rough estimate)", file=sys.stderr)
        return

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
          "env_l11.py — review constrained_llm_l11.py by hand first.")


def run_per_module(args) -> None:
    # In per-module mode each module gets its full .sv without truncation
    # (only one file per prompt so token budget isn't shared).
    max_chars = args.max_chars_per_rtl_file  # user can still override

    prompts = {name: build_module_prompt(name, max_chars, args.coverage_dat)
               for name in RTL_MODULES_OF_INTEREST}
    weights_prompt = build_weights_prompt(args.coverage_dat)

    if args.dry_run:
        for name, prompt in prompts.items():
            print(f"\n{'='*60}\nMODULE: {name}  "
                  f"({len(prompt)} chars, ~{len(prompt)//4} tokens)\n{'='*60}")
            print(prompt)
        print(f"\n{'='*60}\nWEIGHTS PROMPT  "
              f"({len(weights_prompt)} chars, ~{len(weights_prompt)//4} tokens)\n{'='*60}")
        print(weights_prompt)
        total = sum(len(p) for p in prompts.values()) + len(weights_prompt)
        print(f"\n[dry-run] total: {total} chars (~{total//4} tokens) "
              f"across {len(prompts)+1} calls", file=sys.stderr)
        return

    # Fire all module calls in parallel, plus the weights call.
    results: dict[str, str | None] = {}
    weights_response: str | None = None

    all_calls: dict[str, str] = dict(prompts)
    all_calls["__weights__"] = weights_prompt

    print(f"[llm_constraint_synth] firing {len(all_calls)} parallel calls "
          f"to {args.model}...", file=sys.stderr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_calls)) as ex:
        futures = {ex.submit(call_claude, prompt, args.model): name
                   for name, prompt in all_calls.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                text = future.result()
                if name == "__weights__":
                    weights_response = text
                else:
                    results[name] = text
                print(f"[llm_constraint_synth] done: {name}", file=sys.stderr)
            except Exception as exc:
                print(f"[llm_constraint_synth] FAILED: {name}: {exc}", file=sys.stderr)
                if name != "__weights__":
                    results[name] = None

    # Save raw responses.
    out_dir = Path(args.out_response).parent
    for name, resp in results.items():
        if resp:
            (out_dir / f"llm_synth_response_{name}.md").write_text(resp)
    if weights_response:
        (out_dir / "llm_synth_response_weights.md").write_text(weights_response)

    # Merge code blocks.
    parts: list[str] = []

    weights_code = extract_code_block(weights_response or "")
    if weights_code:
        parts.append(f"# --- CATEGORY_WEIGHTS (dedicated weights call) ---\n{weights_code.strip()}")
    else:
        print("[llm_constraint_synth] WARNING: no python block in weights response",
              file=sys.stderr)

    for name in RTL_MODULES_OF_INTEREST:
        resp = results.get(name)
        if not resp:
            continue
        code = extract_code_block(resp)
        if code:
            parts.append(f"# --- {name} ---\n{code.strip()}")
        else:
            print(f"[llm_constraint_synth] WARNING: no python block for {name}",
                  file=sys.stderr)

    if not parts:
        sys.exit("No code blocks extracted from any response — check the raw "
                 "response files in " + str(out_dir))

    header = (
        '"""constrained_llm_l11.py — CRT constraints derived by an LLM ensemble\n'
        "(one API call per RTL module, run in parallel via --per-module mode of\n"
        "llm_constraint_synth.py). CATEGORY_WEIGHTS from a dedicated summary\n"
        "call; build_*_stream functions one per module. UNREVIEWED — read before\n"
        "trusting any RTL claim it makes.\n"
        '"""\n\n'
    )
    merged = "\n\n".join(parts)
    Path(args.out_code).write_text(header + merged + "\n")
    print(f"[llm_constraint_synth] wrote merged constraints -> {args.out_code}")
    print(f"[llm_constraint_synth] {len(parts)} sections merged "
          f"({len(parts)-1} modules + weights).")
    print("[llm_constraint_synth] NOT auto-integrated — review by hand first.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-sonnet-5",
                    help="API model id (default: claude-sonnet-5).")
    ap.add_argument("--max-chars-per-rtl-file", type=int, default=12000,
                    help="Truncation limit per .sv file (default 12000). "
                         "In --per-module mode each module gets its own call "
                         "so you can raise this without multiplying total cost "
                         "— a value of 0 means no truncation.")
    ap.add_argument("--per-module", action="store_true",
                    help="Fire one API call per RTL module in parallel instead "
                         "of a single call with all modules. Each module gets "
                         "its full .sv file (no token budget sharing). "
                         "Results are merged into one constrained_llm_l11.py.")
    ap.add_argument("--out-code", default=str(THIS / "constrained_llm_l11.py"))
    ap.add_argument("--out-response", default=str(THIS / "llm_constraint_synth_response.md"))
    ap.add_argument("--coverage-dat", nargs="*", default=[],
                    help="Path(s) to coverage .dat file(s). Empty by default — "
                         "do NOT pass cpu/coverage_suite_*.dat (lowRISC strategy "
                         "leakage). Only pass .dat from a zero-domain-knowledge run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print prompt(s) without calling the API.")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set.\n\n"
            "This must be an Anthropic Console API key (console.anthropic.com,\n"
            "pay-per-token billing) — separate from a claude.ai Pro/Max chat\n"
            "subscription, which does not by itself grant API access.\n\n"
            "Create a key there, then:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "and re-run this script. Use --dry-run first to preview prompts."
        )

    if args.per_module:
        run_per_module(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
