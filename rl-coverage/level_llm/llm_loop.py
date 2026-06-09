"""llm_loop.py — agentic LLM loop for closing the toggle-coverage gap.

Targets the 121 reachable-but-uncovered toggle bins documented in
accessible_bins_for_llm.txt (8 groups). For each group:

  1. Build a prompt: ISA encoding cheat sheet + group explanation + target bins
  2. Ask the LLM (Groq / Llama-3.3-70B) for a short instruction sequence,
     expressed directly in the (op, rd, rs1, rs2, imm_bucket) action format
  3. Run it on real Verilator RTL via starter.run_and_check — this is the
     "deterministic evaluator": ground truth comes from execution, not from
     the LLM's self-assessment
  4. Feed back EXACTLY which target signals toggled and which didn't; ask
     for a refined attempt that builds on what worked
  5. Repeat up to MAX_ROUNDS per group; log every execution-validated success

After the per-group passes, bins that are still uncovered are pooled into a
"worst-state" round: the hardest residue gets focused, individually-targeted
attempts with a larger round budget — LLM4Cov's worst-state-prioritized
sampling, applied to where to spend the (rate-limited) generation budget.

No model gets trained: at this scale (121 bins) one capable model running the
agentic loop directly against the execution-validated evaluator is the right
amount of machinery — see conversation notes for why a teacher->student
distillation pipeline would be solving a problem this project doesn't have.
"""
import os, re, sys, json, time, argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

import starter
import isa_reference as ref

MODEL        = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_ROUNDS   = 4
WORST_ROUNDS = 6
RESULTS_PATH = THIS / "agentic_results.json"
BINS_FILE    = THIS / "accessible_bins_for_llm.txt"

_client = None
def client():
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            sys.exit("GROQ_API_KEY not set — put it in level_llm/.env (see .gitignore)")
        _client = Groq(api_key=key)
    return _client


# ── Parse the 8 groups straight out of accessible_bins_for_llm.txt ───────────
GROUP_RE = re.compile(r"GROUP (\d+): (.+?) \((\d+) bins\)")
SIGNAL_LINE_RE = re.compile(r"^\s{2}(\S+)\s{2,}(\S+)\s*$")

def parse_groups(path=BINS_FILE):
    text = path.read_text()
    blocks = re.split(r"={20,}\nGROUP", text)[1:]
    groups = []
    for block in blocks:
        block = "GROUP" + block
        m = GROUP_RE.search(block)
        if not m:
            continue
        gid, title, n = int(m.group(1)), m.group(2).strip(), int(m.group(3))
        expl_m = re.search(r"EXPLANATION:\n(.*?)\n\n", block, re.S)
        sig_m  = re.search(r"SIGNALS:\n(.*?)\n\n", block, re.S)
        prog_m = re.search(r"SUGGESTED PROGRAM:\n(.*?)(?:\n=|\Z)", block, re.S)
        signal_names = []
        if sig_m:
            for line in sig_m.group(1).splitlines():
                sm = SIGNAL_LINE_RE.match(line)
                if sm:
                    signal_names.append(sm.group(2))
        groups.append({
            "id": gid, "title": title, "n_bins": n,
            "explanation": (expl_m.group(1).strip() if expl_m else ""),
            "signal_names": signal_names,
            "suggested_program": (prog_m.group(1).strip() if prog_m else "(none provided)"),
        })
    return groups


def relevant_csrs_for_group(group):
    """Only show CSR routing table entries the group's text actually mentions
    — keeps the prompt (and the rate-limited token budget) small."""
    haystack = (group["explanation"] + " " + group["suggested_program"] + " "
                + " ".join(group["signal_names"])).lower()
    found = [csr for csr, name in ref.CSR_NAMES.items() if name.lower() in haystack]
    return found or None


# ── Prompting ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a hardware verification engineer generating short RISC-V "
    "RV32IMC test programs for an Ibex CPU running in Verilator simulation "
    "(M-mode only). Your programs are NOT free-form assembly — every "
    "instruction must be expressed as a 5-field action object that gets "
    "encoded by a fixed hardware codec. Read the encoding reference "
    "carefully: many fields are quantized to a small table of pre-baked "
    "values, not arbitrary numbers. Respond with ONLY a JSON array, no prose."
)

OUTPUT_FORMAT = (
    'Respond with ONLY a JSON array of action objects, 5-40 instructions long, '
    'e.g.:\n'
    '[{"op": "LUI", "rd": 5, "rs1": 0, "rs2": 0, "imm_bucket": 3},\n'
    ' {"op": "CSRRW", "rd": 0, "rs1": 5, "rs2": 0, "imm_bucket": 1}]\n'
    'Every object needs "op" (a name from the OP TABLE) and integer '
    '"rd","rs1","rs2","imm_bucket" (any field you omit defaults to 0). '
    'No markdown fences, no commentary — JSON array only.'
)


def build_prompt(group, history=None, focus_signals=None):
    target = focus_signals if focus_signals else group["signal_names"]
    sheet = ref.cheat_sheet_text(target_csrs=relevant_csrs_for_group(group))
    bins_txt = "\n".join(f"  - {s}" for s in target)

    parts = [
        f"GOAL: write ONE short instruction sequence that makes EACH of these "
        f"toggle-coverage signals transition both 0→1 AND 1→0 "
        f"(a signal counts as covered only once BOTH transitions are observed "
        f"during the run):\n{bins_txt}\n",
        f"WHY these are hard / what hardware path is involved:\n{group['explanation']}\n",
        f"A prior (unvalidated) sketch suggested this assembly as a starting "
        f"point. It is NOT directly emittable in our action format (free "
        f"immediates, labels, pseudo-ops) — translate the INTENT, not the "
        f"syntax:\n{group['suggested_program']}\n",
        sheet,
        OUTPUT_FORMAT,
    ]
    if history:
        fb = ["EXECUTION-VALIDATED FEEDBACK FROM PREVIOUS ATTEMPTS (real RTL "
              "simulation — this is ground truth, trust it over your own "
              "reasoning about the hardware):"]
        for i, h in enumerate(history, 1):
            if h.get("error"):
                fb.append(f"  Attempt {i}: REJECTED — {h['error']}")
                continue
            fb.append(f"  Attempt {i}: {json.dumps(h['program'])}")
            fb.append(f"    -> newly toggled (both 0→1 and 1→0): {sorted(h['hit'])}")
            fb.append(f"    -> still NOT toggled: {sorted(h['missing'])}")
        fb.append("\nWrite a NEW, refined sequence. Keep what worked; change "
                   "your approach only for the signals still listed as not "
                   "toggled — they need a different instruction, operand, or "
                   "ordering than what you tried.")
        parts.append("\n".join(fb))
    return "\n\n".join(parts)


def extract_json_array(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        elif start != -1:
            text = text[start:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Response got cut off mid-array (hit max_tokens) — salvage the
        # complete action objects generated before the cutoff rather than
        # discarding the whole attempt.
        last = text.rfind("}")
        if last == -1:
            raise
        return json.loads(text[:last + 1].rstrip().rstrip(",") + "]")


def call_llm(prompt, retries=4):
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"    [llm error: {e!r}; retrying in {wait}s]")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")


# ── One agentic attempt: LLM proposes -> Verilator validates ─────────────────
def attempt(prompt, baseline_ids, covdat_map, target_set):
    raw = call_llm(prompt)
    try:
        actions_json = extract_json_array(raw)
        program = ref.parse_program(actions_json)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"    [parse error: {e}]")
        return {"program": [], "hit": set(), "missing": set(target_set), "error": str(e)}

    print(f"    [program: {len(program)} instructions]")
    csr_ops = [a for a in actions_json if a.get("op","").upper() in ("CSRRW","CSRRS","CSRRC","CSRRWI","CSRRSI","CSRRCI")]
    print(f"    [CSR ops in program: {csr_ops[:5]}]")
    result = starter.run_and_check(program, baseline_ids, covdat_map, target_set, debug_signal="mcountinhibit")
    print(f"    [new bins total: {result['new_total']}  target: {result['new_target']}  total_covered: {result.get('total_covered', '?')}  signals: {result['signals'][:5]}{'...' if len(result['signals']) > 5 else ''}]")
    hit = set(result["target_signals"]) & target_set
    return {"program": actions_json, "hit": hit, "missing": target_set - hit,
            "all_new_signals": result["signals"]}


# ── Per-group agentic loop ────────────────────────────────────────────────────
def run_group(group, baseline_ids, covdat_map, max_rounds=MAX_ROUNDS):
    target_set = set(group["signal_names"])
    solved, history = set(), []
    print(f"\n=== GROUP {group['id']}: {group['title']} ({len(target_set)} bins) ===")
    for rnd in range(1, max_rounds + 1):
        prompt = build_prompt(group, history=history)
        h = attempt(prompt, baseline_ids, covdat_map, target_set - solved)
        solved |= h["hit"]
        h["missing"] = target_set - solved
        history.append(h)
        status = h.get("error", f"+{len(h['hit'])} new")
        print(f"  round {rnd}: {status} -> {len(solved)}/{len(target_set)} solved")
        if not h["missing"]:
            break
    return solved, history


# ── Worst-state pass: pool everything still missing, attack it focused ───────
def run_worst_state(remaining, baseline_ids, covdat_map, group_by_signal,
                    max_rounds=WORST_ROUNDS):
    if not remaining:
        return set(), []
    print(f"\n=== WORST-STATE PASS: {len(remaining)} bins resisted the group passes ===")
    pseudo_group = {
        "id": "worst", "title": "worst-state residue",
        "explanation": (
            "These signals survived a full pass of group-level attempts. "
            "They likely need a SPECIFIC instruction sequence (operand "
            "ordering, a particular register value, a multi-step setup) "
            "rather than a broad mix. Think step by step about the exact "
            "hardware path for each signal before writing the sequence."),
        "suggested_program": "(none — these resisted the suggested sketches too)",
        "signal_names": sorted(remaining),
    }
    solved, history = set(), []
    for rnd in range(1, max_rounds + 1):
        target_set = set(remaining) - solved
        if not target_set:
            break
        prompt = build_prompt(pseudo_group, history=history, focus_signals=sorted(target_set))
        h = attempt(prompt, baseline_ids, covdat_map, target_set)
        solved |= h["hit"]
        h["missing"] = target_set - h["hit"]
        history.append(h)
        status = h.get("error", f"+{len(h['hit'])} new")
        print(f"  round {rnd}: {status} -> {len(solved)}/{len(remaining)} solved")
        if not h["missing"]:
            break
    return solved, history


# ── Driver ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=None,
                    help="run only this group id (1-8); default: all groups + worst-state pass")
    ap.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--no-worst-state", action="store_true")
    ap.add_argument("--list", action="store_true", help="list parsed groups and exit")
    args = ap.parse_args()

    groups = parse_groups()
    if args.list:
        for g in groups:
            print(f"GROUP {g['id']}: {g['title']} — {len(g['signal_names'])} signals parsed "
                  f"(file says {g['n_bins']})")
        return

    baseline_ids, covdat_map, _ = starter.setup()

    targets = [g for g in groups if (args.group is None or g["id"] == args.group)]
    all_solved, group_results = {}, {}
    group_by_signal = {}
    for g in groups:
        for s in g["signal_names"]:
            group_by_signal[s] = g["id"]

    t0 = time.time()
    for g in targets:
        solved, history = run_group(g, baseline_ids, covdat_map, max_rounds=args.rounds)
        all_solved[g["id"]] = solved
        group_results[g["id"]] = history

    if args.group is None and not args.no_worst_state:
        remaining = set()
        for g in groups:
            remaining |= (set(g["signal_names"]) - all_solved.get(g["id"], set()))
        ws_solved, ws_history = run_worst_state(remaining, baseline_ids, covdat_map, group_by_signal)
        group_results["worst_state"] = ws_history
        for s in ws_solved:
            all_solved.setdefault(group_by_signal.get(s, "worst"), set()).add(s)

    elapsed = time.time() - t0
    total_target = sum(len(g["signal_names"]) for g in targets)
    total_solved = sum(len(v) for k, v in all_solved.items()
                       if any(g["id"] == k for g in targets) or k == "worst")
    print(f"\n{'='*60}\nDONE in {elapsed/60:.1f} min — "
          f"{total_solved}/{total_target} target bins execution-validated as covered\n{'='*60}")

    # ── persist results: solved bins + winning programs, for reuse ───────────
    out = {"model": MODEL, "elapsed_s": elapsed, "groups": {}}
    for g in groups:
        gid = g["id"]
        history = group_results.get(gid, [])
        winning = [h for h in history if h.get("hit")]
        out["groups"][gid] = {
            "title": g["title"],
            "targets": g["signal_names"],
            "solved": sorted(all_solved.get(gid, set())),
            "winning_programs": [{"program": h["program"], "newly_hit": sorted(h["hit"])}
                                 for h in winning],
        }
    if "worst_state" in group_results:
        winning = [h for h in group_results["worst_state"] if h.get("hit")]
        out["worst_state"] = {
            "winning_programs": [{"program": h["program"], "newly_hit": sorted(h["hit"])}
                                 for h in winning],
        }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
