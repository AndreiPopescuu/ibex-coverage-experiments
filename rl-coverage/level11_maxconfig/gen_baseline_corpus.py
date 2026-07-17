"""gen_baseline_corpus.py — freeze a deterministic stimulus program covering
all N_OPS=143 codec_l11 ops (RV32IMC + RV32B Zba/Zbb/Zbs + 27 legacy-draft
zbp/zbc/zbe/zbf ops), for use as a fixed old-vendor-vs-upstream-vendor
toggle-coverage comparison corpus.

Writes corpus_max_baseline_v1.json (checked in — unlike the gitignored
coverage .dat/.pkl outputs, this needs to persist so the comparison is
reproducible later).
"""
import json
import os

from codec_l11 import N_OPS, IMM_BUCKETS, N_CSR_BUCKETS, emit_program

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "corpus_max_baseline_v1.json")

# Registers to cycle through (avoid x0 as rd for most variants so results are
# observable; a few variants deliberately use x0/matching rs1==rs2 to also
# exercise the ZERO_DST / SAME_SRC-style decoder paths other levels track).
REG_CYCLE = list(range(1, 32))


def _variants_for_op(op_i, n=4):
    out = []
    for k in range(n):
        rd = REG_CYCLE[(op_i * 7 + k * 3) % len(REG_CYCLE)]
        rs1 = REG_CYCLE[(op_i * 11 + k * 5) % len(REG_CYCLE)]
        rs2 = REG_CYCLE[(op_i * 13 + k * 7 + 1) % len(REG_CYCLE)]
        imm_bucket = (op_i + k) % IMM_BUCKETS
        csr_bucket = (op_i + k) % max(N_CSR_BUCKETS, 1)
        out.append((op_i, rd, rs1, rs2, imm_bucket, csr_bucket))
    # A couple of edge-case variants: rd=x0 (ZERO_DST-style), rs1==rs2 (SAME_SRC-style)
    out.append((op_i, 0, REG_CYCLE[op_i % len(REG_CYCLE)], REG_CYCLE[(op_i + 1) % len(REG_CYCLE)],
                op_i % IMM_BUCKETS, op_i % max(N_CSR_BUCKETS, 1)))
    r = REG_CYCLE[op_i % len(REG_CYCLE)]
    out.append((op_i, REG_CYCLE[(op_i + 2) % len(REG_CYCLE)], r, r,
                op_i % IMM_BUCKETS, op_i % max(N_CSR_BUCKETS, 1)))
    return out


def build_actions():
    actions = []
    for op_i in range(N_OPS):
        actions.extend(_variants_for_op(op_i))
    return actions


def main():
    actions = build_actions()
    machine_code = emit_program(actions)
    payload = {
        "n": len(machine_code),
        "n_ops": N_OPS,
        "n_actions": len(actions),
        "agent": "baseline_v1_all_ops",
        "machine_code": machine_code,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {OUT_PATH}: {len(actions)} actions -> {len(machine_code)} words "
          f"(covers all {N_OPS} codec_l11 ops)")


if __name__ == "__main__":
    main()
