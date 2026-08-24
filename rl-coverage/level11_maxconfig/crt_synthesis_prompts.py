"""crt_synthesis_prompts.py — the FIXED prompt templates used to brief the
isolated agents that write/review new constrained_llm_l11.py streams.

Why this file exists: passes 1-2 (9->13 modules, 40->63 streams) never saved
the exact prompt text anywhere -- it was typed fresh, by hand, each time,
directly into a Claude Code session, and is now unrecoverable. Pass 3 (today,
ibex_core/ibex_decoder, 63->68 streams) used the text captured here verbatim.
From this point on, EVERY future round should build its prompt through this
file, not by re-typing it, so the wording is provably identical across
rounds and the prompt itself is never lost again.

The two templates below are FIXED (do not edit the wording casually -- if
you change it, you're changing the methodology, not just formatting). Only
the data plugged into the placeholders varies per round: which RTL file,
which uncovered signals, which draft report(s) to review.

The "action space" section is NOT duplicated here -- it's pulled live from
llm_constraint_synth.py's gather_action_space_summary(), which derives it
straight from codec_l11.py's real L11_CSRS pool. That way this file can
never drift out of sync with the actual codec as it grows.

Usage:
    # print a synthesis-agent prompt ready to paste into Agent(...)'s prompt:
    python3 crt_synthesis_prompts.py synthesis \\
        --rtl-path cpu/src_upstream/lowrisc_ibex_ibex_core_0.1/rtl/ibex_icache.sv \\
        --signals "data_tweak_lw_ic0 (78 bits missing)" "fill_ram_req_addr (21 bits missing)"

    # print a review-agent prompt, embedding one or more draft reports:
    python3 crt_synthesis_prompts.py review \\
        --rtl-path cpu/src_upstream/.../ibex_icache.sv \\
        --draft-file /path/to/draft1.md [--draft-file /path/to/draft2.md ...]

Both subcommands print the finished prompt to stdout -- pipe it into a
clipboard tool, or copy it straight into an Agent(...) call's `prompt` field.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_constraint_synth import gather_action_space_summary  # noqa: E402

# The exact example function shown in every Pass 3 synthesis prompt, kept
# verbatim as the canonical style/format reference -- this is a code-
# convention example ONLY, not RTL knowledge, which is why reusing it across
# every future round doesn't violate the "zero borrowed knowledge" rule.
EXAMPLE_FUNCTION = '''def build_core_illegal_insn_toggle_stream(rng):
    """Target ibex_core.sv:391,723 -- illegal_insn_id / unused_illegal_insn_id wires.

    ibex_core.sv declares (line 391) `logic illegal_insn_id,
    unused_illegal_insn_id;` and (line 723, unconditional, not inside any
    ifdef) `assign unused_illegal_insn_id = illegal_insn_id;`.
    illegal_insn_id is wired straight through from ibex_id_stage's
    .illegal_insn_o. This is the only elaborated consumer of illegal_insn_id
    in ibex_core.sv (the fcov block that also reads it is stripped under
    -DSYNTHESIS=1, per cpu/Makefile.upstream).

    ILLEGAL_INSN (op=83, codec_l11.py) is a fixed CUSTOM-0 encoding
    (0x0000000B) that falls through the decoder's default case to
    illegal_insn_o=1. Alternating it 1:1 against a cheap legal instruction
    (ADDI, op=10) forces a 0->1->0 transition on illegal_insn_id every 2
    cycles.
    """
    stream = []
    for _ in range(24):
        stream.append((83, 0, 0, 0, 2, 0))  # ILLEGAL_INSN -> illegal_insn_id = 1
        stream.append((10, rng.randint(1, 31), rng.randint(1, 31), 0,
                       rng.randint(0, 4), 0))  # ADDI (legal) -> illegal_insn_id = 0
    return stream'''

SYNTHESIS_PROMPT_TEMPLATE = """You are deriving constrained-random test-generation rules for an \
Ibex RISC-V CPU core, targeting RTL toggle-coverage in a Verilator testbench. This is a \
research project measuring whether an LLM given ONLY RTL source and this project's own \
coverage feedback can derive test constraints that perform comparably to a hand-built \
expert test suite — WITHOUT using any borrowed verification knowledge.

IMPORTANT: Do NOT reference, assume, or rely on lowRISC's own verification IP (riscv-dv, \
UVM testbenches, testlist.yaml, or any of their existing test names/strategies), and do \
not use general RISC-V-verification-methodology knowledge from elsewhere. Derive \
everything purely from reading the RTL file below and the coverage/action-space \
information given here. The goal is an INDEPENDENT derivation.

## Your task

Read this RTL file yourself with the Read tool (do not ask for it to be pasted):

  {rtl_file_path}

Use Grep on this same file to find exactly where these signals are assigned (these are \
Verilator toggle-coverage bins left UNCOVERED by a prior constrained-random test run — \
i.e. these bits never changed value across the whole run):

{uncovered_signals_block}

For each signal, find the exact assignment/case-arm/mux driving it and cite the file:line \
you're relying on.

## The ONLY action space you may target with (a stimulus generator, not a real assembler)

{action_space_summary}

## Output contract

One or more Python functions, each `build_<name>_stream(rng) -> list[tuple[int,int,int,\
int,int,int]]` where each tuple is (op, rd, rs1, rs2, imm_bucket, csr_bucket), each \
targeting ONE specific uncovered signal/structure you identified above. Each function \
needs a docstring citing the exact RTL line number(s) and signal names you're relying on, \
and explaining WHY the action sequence should toggle bits that weren't toggling before.

Here is the exact style/format this project's existing stream functions use (for \
interface/formatting reference ONLY — this is this project's own code convention, not \
RTL knowledge, feel free to match this shape):

```python
{example_function}
```

Output ONLY a single ```python fenced code block containing your function(s), plus a \
short prose section (outside the code block) explaining your reasoning per stream, citing \
exact RTL lines. Do not integrate into any other file — just produce the code block and \
explanation, someone else will merge it. If you conclude a target signal genuinely cannot \
be reached given this action space's hard constraints, say so explicitly instead of \
inventing something that wouldn't actually work — report that as a finding, don't force a \
fake stream for it.
"""

REVIEW_PROMPT_TEMPLATE = """You are the independent adversarial review pass for the \
draft report(s) below, produced by another agent (or agents) that derived constrained-\
random test-generation stream builders for an Ibex RISC-V CPU core (Verilator toggle-\
coverage project). Your job: independently re-derive and cross-check every RTL claim in \
the draft(s) — do NOT rubber-stamp them. You have NO context from whatever conversation \
produced these drafts; treat them as unverified external submissions.

## Background you need

This project drives Ibex through a flat instruction stream encoded by a Python codec \
(NOT a real assembler) — every "instruction" is a 6-tuple action `(op, rd, rs1, rs2, \
imm_bucket, csr_bucket)` passed to `encode()`. The authoritative encoder logic lives in \
these files (READ THEM YOURSELF to verify every op-index and CSR claim below — do not \
trust the drafts' arithmetic):

  rl-coverage/level11_maxconfig/codec_l11.py
  rl-coverage/level10_ops/codec_l10.py   (defines the `Op` IntEnum with all base op \
indices 0-86, and the CSR pool L10_CSRS)

The RTL file(s) the draft(s) are targeting (READ THESE YOURSELF too — every specific \
line-number/signal claim below must be checked against the actual current file contents, \
not assumed correct):

{rtl_file_paths_block}

Existing convention for stream functions already in the project (check the actual style \
already in the file too):

  rl-coverage/level11_maxconfig/constrained_llm_l11.py

## What to check, specifically

For EVERY function in the draft(s):
1. Does the cited RTL line number / signal name actually exist and say what the draft \
claims, in the CURRENT file? RTL may have shifted since the draft was written — re-find \
the real line numbers.
2. Is every op index used actually correct per `codec_l10.py`'s `Op` IntEnum? Don't trust \
the comment next to the number — check the enum yourself.
3. For any CSR the draft references by name with a placeholder `csr_bucket`, verify that \
CSR actually exists in `L10_CSRS`/`L11_CSRS` and is NOT gated behind `debug_mode_i` or \
any other precondition that would make the write silently fail from ordinary M-mode code.
4. For every claim that a signal is "structurally unreachable" given this codec/action-\
space, independently verify that claim by reading the RTL yourself — don't just trust the \
draft's reasoning. If you find a way it COULD be reached that the draft missed, say so \
and propose a stream. If you confirm it's unreachable, say so explicitly with your own \
citation.
5. Check for basic Python correctness: tuple arity (must be exactly 6 elements matching \
(op,rd,rs1,rs2,imm_bucket,csr_bucket)), rd/rs1/rs2 always in range 0-31, imm_bucket \
always in range 0-4, csr_bucket always a valid index or clearly-marked placeholder.

## Draft(s) to review

{draft_reports_block}

## Output

For each proposed stream function: state CONFIRMED (all citations check out, safe to \
merge as-is), NEEDS FIX (say exactly what's wrong and what the fix is — give corrected \
code if you can), or REJECT (fundamentally broken/unreachable despite the draft's claim). \
For each "unreachable" finding: state CONFIRMED UNREACHABLE (you independently verified \
the same conclusion, cite your own line numbers) or DISPUTE (you found a way it IS \
reachable — propose a stream). Be thorough and skeptical — this project's whole \
methodology depends on these RTL claims actually being true, not just plausible-sounding. \
End with a final corrected Python code block containing only the functions that are \
CONFIRMED or NEEDS FIX-and-now-fixed, ready to merge as-is into constrained_llm_l11.py.
"""


def build_synthesis_prompt(rtl_file_path: str, uncovered_signals: list[str]) -> str:
    signals_block = "\n".join(f"  {s}" for s in uncovered_signals)
    return SYNTHESIS_PROMPT_TEMPLATE.format(
        rtl_file_path=rtl_file_path,
        uncovered_signals_block=signals_block,
        action_space_summary=gather_action_space_summary(),
        example_function=EXAMPLE_FUNCTION,
    )


def build_review_prompt(rtl_file_paths: list[str], draft_texts: list[str]) -> str:
    rtl_block = "\n".join(f"  {p}" for p in rtl_file_paths)
    drafts_block = "\n\n---\n\n".join(draft_texts)
    return REVIEW_PROMPT_TEMPLATE.format(
        rtl_file_paths_block=rtl_block,
        draft_reports_block=drafts_block,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_syn = sub.add_parser("synthesis", help="print a synthesis-agent prompt")
    p_syn.add_argument("--rtl-path", required=True,
                        help="path to the target RTL file, e.g. cpu/src_upstream/.../ibex_icache.sv")
    p_syn.add_argument("--signals", nargs="+", required=True,
                        help="uncovered signal descriptions, one per --signals arg, "
                             "e.g. \"data_tweak_lw_ic0 (78 bits missing)\"")

    p_rev = sub.add_parser("review", help="print a review-agent prompt")
    p_rev.add_argument("--rtl-path", action="append", required=True, dest="rtl_paths",
                        help="target RTL file path; repeat for multiple files")
    p_rev.add_argument("--draft-file", action="append", required=True, dest="draft_files",
                        help="path to a draft synthesis-agent report to embed; repeat for multiple drafts")

    args = ap.parse_args()

    if args.cmd == "synthesis":
        print(build_synthesis_prompt(args.rtl_path, args.signals))
    elif args.cmd == "review":
        draft_texts = [Path(f).read_text() for f in args.draft_files]
        print(build_review_prompt(args.rtl_paths, draft_texts))


if __name__ == "__main__":
    main()
