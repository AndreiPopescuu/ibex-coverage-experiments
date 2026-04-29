# Level 8 — Directed + RL for LINE Coverage

## TL;DR

Target: **≥85% line coverage** on minimal Ibex (vs. ~20% current baseline).  
Stretch goal: **~92%** (estimated reachable ceiling with minimal config).

| Approach | Expected result | Time |
|---|---|---|
| `directed_l8.py` (one directed run) | 70–80% line | ~2 min |
| `greedy_l8.py` (30 random episodes) | 75–82% line | ~1 h |
| `train_l8.py` (200 PPO episodes)    | 82–90% line | ~3–5 h |
| `train_l8.py --directed-first`      | **85–92% line** | ~3–5 h |

---

## Why we were stuck at 20%

The 20% baseline came from an action space that mapped to a small instruction
subset (R-type + stores + JAL).  Every RTL module with a `case` statement for
instruction types — especially `ibex_tracer` (488 lines, 39% of all line
points!) — had most of its arms never hit.

Six instruction types were **completely missing** from L7's codec:
`LUI`, `JALR`, `CSRRWI`, `CSRRSI`, `CSRRCI`, `FENCE`.
Each one corresponds to at least one uncovered `case` arm in the tracer and decoder.

Additionally, branches were only emitted in one direction (never both
taken **and** not-taken in the same episode), leaving branch-path RTL lines
uncovered.

---

## What's new in Level 8

### codec_l8.py — 70 ops (L7 + 6)

| New op | opcode | Why it matters |
|---|---|---|
| `LUI`    | `0110111` | ibex_tracer LUI arm + upper-imm ALU path |
| `JALR`   | `1100111` | ibex_tracer JALR arm + indirect-branch controller path |
| `CSRRWI` | `1110011` f3=101 | tracer CSRRWI arm + cs_registers immediate-write path |
| `CSRRSI` | `1110011` f3=110 | tracer CSRRSI arm |
| `CSRRCI` | `1110011` f3=111 | tracer CSRRCI arm |
| `FENCE`  | `0001111` | tracer MISC-MEM arm + decoder fence path |

### directed_l8.py — Comprehensive directed program

A ~2000-instruction program covering:
- All 10 R-type ALU ops with 5 operand variants each (normal, rd=x0, rs1=x0, same-src, extreme values)
- All 8 M-extension ops (MUL/DIV) with `0`, `-1`, `INT_MAX`, `INT_MIN` operands
- All 9 I-type immediates (ADDI/SLTI/SLTIU/XORI/ORI/ANDI/SLLI/SRLI/SRAI)
- LUI + AUIPC with 6 different upper-immediate buckets
- All load widths (LB/LBU/LH/LHU/LW) with stores first
- JAL: forward + backward (non-looping pattern from test_cpu_coverage)
- JALR: subroutine call/return pattern
- **Branches: ALL 6 types × BOTH taken and not-taken** (BEQ/BNE/BLT/BGE/BLTU/BGEU)
- CSR ops: all 6 variants × 12 safe CSR addresses
- ECALL + EBREAK × 2 (exercises controller exception FSM re-entry)
- All 16 compressed instruction types
- FENCE
- Large-operand MUL/DIV for multi-cycle state machine coverage

### test_run_for_l8.py — Enhanced cocotb driver

- Prologue initialises **all 32 registers** with diverse values:
  - x16 = 0xFFFFFFFF (-1 for signed comparisons)
  - x17 = 0x7FFFFFFF (INT_MAX)
  - x18 = 0x80000000 (INT_MIN)
  - x19–x23 = alternating-bit patterns, large hex constants
- Trap handler reads `mepc` + `mcause` + `mtval` before MRET (more CSR paths)
- Data memory: `addr XOR 0xDEADBEEF` for read-misses (from L7)
- `max_cycles = len(program) * 15 + 1000` (extra headroom for MUL/DIV + traps)

### env_l8.py — Line-coverage gym env

```python
env = IbexL8Env(
    kind         = "line",      # optimise line coverage (not toggle)
    reward_mode  = "compound",  # new_hits*10 + ep_line_pct*0.5
    episode_steps = 1024,
)
```

**Reward formula (compound)**:
```
reward = new_line_hits * 10  +  ep_line_pct * 0.5
```
- `new_line_hits * 10`: strong gradient toward discovering new RTL lines
- `ep_line_pct * 0.5`: keeps agent from drifting away from already-covered areas

**Observation** (4 floats):
```
[step / episode_steps,  ep / 200,  cum_line_pct,  ep / 100]
```

---

## How to run

```bash
# Step 0: build Vtop (if not already done)
cd ../../cpu && make && cd -

# Step 1: DIRECTED run — biggest single gain, ~2 min
python directed_l8.py

# Step 2a: Random baseline (compare against PPO)
python greedy_l8.py --episodes 30

# Step 2b: PPO training
python train_l8.py --episodes 200 --reward compound

# Step 3: Combined (directed seed + PPO)
python train_l8.py --episodes 200 --reward compound   # already runs directed first

# Step 4: Inspect what's still missing
python plot_l8.py --annotate

# Step 5: Annotate source files to see uncovered lines
verilator_coverage --annotate /tmp/ibex_l8_annotate ../../cpu/coverage.dat
# Then look at e.g.:
#   grep -n "%000000" /tmp/ibex_l8_annotate/ibex_tracer.sv | head -20
```

---

## Expected per-module improvement

| Module | Before L8 | After directed | After PPO |
|---|---|---|---|
| ibex_tracer (488 lines) | ~5% | ~75% | ~85% |
| ibex_decoder (197 lines) | ~15% | ~70% | ~80% |
| ibex_cs_registers (101) | ~8% | ~65% | ~75% |
| ibex_load_store_unit (68) | ~20% | ~80% | ~90% |
| ibex_alu (66) | ~25% | ~80% | ~88% |
| ibex_multdiv_fast (19) | ~15% | ~70% | ~85% |
| ibex_compressed_decoder (34) | ~5% | ~75% | ~85% |
| ibex_controller (37) | ~10% | ~50% | ~65% |
| prim_secded_pkg (76) | 0% | 0% | 0% (unreachable) |
| prim_cipher_pkg (35) | 0% | 0% | 0% (unreachable) |

---

## Reachable ceiling analysis

With minimal Ibex config (PMPEnable=0, ICache=0, DbgTriggerEn=0,
SecureIbex=0, WritebackStage=0), approximately **111 lines are
structurally unreachable**:

- `prim_secded_pkg`: 76 lines — ECC encode/decode, not instantiated
- `prim_cipher_pkg`: 35 lines — PRINCE/PRESENT cipher, only used for
  scrambled iCache (which is disabled)

Reachable ceiling: **(1241 - 111) / 1241 = 91.1%**

To reach **95%** (the lowRISC benchmark target):
→ Enable opentitan config: `PMPEnable=1`, `ICache=1`, `DbgTriggerEn=1`,
  `SecureIbex=1`, `WritebackStage=1` in `cpu/cocotb_ibex.sv`.
  This expands the surface from 1241 to ~5000–8000 line points but makes
  all modules reachable, enabling 94.8%+ with directed stimulus.

---

## Files

| File | Purpose |
|---|---|
| `codec_l8.py` | 70-op encoder (L7 + LUI/JALR/CSRRWI/SI/CI/FENCE) |
| `directed_l8.py` | Comprehensive directed program builder (~2000 insns) |
| `env_l8.py` | Gym env: `kind="line"`, compound reward |
| `train_l8.py` | PPO training with SB3 |
| `greedy_l8.py` | Random accumulation baseline |
| `plot_l8.py` | Coverage curve charts + per-module annotation |
| `../../cpu/test_run_for_l8.py` | Cocotb driver with 32-reg init + extended trap handler |
