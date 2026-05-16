# Shadow-Based RL Coverage Testing for Ibex RISC-V

## Overview

This work explores using Reinforcement Learning (RL) with a Python shadow simulator to achieve functional coverage of the Ibex RISC-V processor, targeting the Rich Functional Coverage model (~15,217 bins across 9 categories: SEEN, OPERAND_PAIR, RAW_HAZARD, SEQUENCE, CORNER_CASES, etc.).

**Baseline to beat:** Constrained Random (480 steps, 300 episodes) → **45.10%**

---

## 1. Python Shadow Simulator

A complete RISC-V ISA simulator (~400 lines) was implemented to replace Verilator during RL training:

- All 70 opcodes supported (R-type, I-type, loads/stores, branches, JAL/JALR, compressed, CSR)
- Register file initialized identically to Verilator's test prologue (x16 = -1, x17 ≈ INT_MAX, x18 = INT_MIN, etc.)
- Memory model matches DiverseMemAgent: unknown addresses return addr ^ 0xDEADBEEF
- **Speed: ~2s/episode vs ~30s/episode with Verilator (15× faster)**

---

## 2. Critical Bug: Control Flow Execution

**Problem discovered:** The shadow executed instructions sequentially (instruction[0], instruction[1], ..., instruction[N]) ignoring branches and jumps, while Verilator follows the actual program counter.

**Impact:** Shadow reported 3× more coverage bins than Verilator on identical action sequences.

**Root cause breakdown (5 random episodes):**

| Category | Only Shadow | Only Vtop | Both |
|----------|------------|-----------|------|
| seq_ (SEQUENCE) | 554 | 104 | 301 |
| opair_ (OPERAND_PAIR) | 324 | 35 | 217 |
| raw (RAW_HAZARD) | 58 | 11 | 22 |

**Fix:** Implemented run_machine_code_cf() — control-flow-aware execution that maps PC delta after each instruction to the correct next instruction slot.

**Result after fix:**

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Shadow cumulative % | 11.30% | 5.89% |
| Verilator cumulative % | 5.88% | 5.88% |
| Divergence ratio | 3× | ~1.19× |

---

## 3. Diagnostic Methodology

A per-category comparison script was developed to identify divergence sources between shadow and Verilator on identical programs:
- Classifies hits into: only_shadow, only_vtop, both
- Breaks down by bin category (SEEN, OPERAND_PAIR, RAW_HAZARD, SEQUENCE, CORNER)
- After the fix, divergence became symmetric (~178 bins each side), confirming accuracy

---

## 4. Improved Gym Environment

**Observation space expanded from 6 → 12 features** (the original had 3 features permanently set to zero):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | step_frac | Position within episode |
| 1 | cum_cov | Global cumulative coverage fraction |
| 2 | seen_frac | SEEN bins covered |
| 3 | opair_frac | OPERAND_PAIR bins covered |
| 4 | raw_frac | RAW_HAZARD bins covered |
| 5 | seq_frac | SEQUENCE bins covered |
| 6 | corner_frac | CORNER_CASES bins covered |
| 7 | ep_new_frac | New bins found this episode |
| 8 | last_op_norm | Last opcode (normalized) |
| 9 | ep_idx_norm | Episode index (normalized) |
| 10 | last_rd | Last destination register written |
| 11 | raw_potential | Recent producer instructions in window |

**Per-step structural reward** added for RAW hazard patterns: when the agent selects rs1/rs2 matching a recently written rd, a small reward (+2.0) is given immediately — teaching RAW patterns without hardcoding them.

---

## 5. Infrastructure Fix

Verilator subprocess was failing due to cocotb not finding the correct Python interpreter. Fixed by injecting VIRTUAL_ENV and prepending the venv bin/ to PATH in the subprocess environment.

---

## 6. Training Scale Comparison

| Approach | Episodes | Steps/ep | Wall Time | Verilator Coverage |
|----------|---------|----------|-----------|-------------------|
| PPO on Verilator (original) | 300 | 256 | ~2.5h | 36.08% |
| Constrained Random | 300 | 480 | ~2.5h | 45.10% |
| PPO on Shadow (fixed) | 2000 | 512→1024 | ~100 min | ~35% vtop |
| PPO on Shadow (current) | 2000 | 480 fixed | ~67 min | TBD |

---

## 7. Key Finding: Sim-to-Real Gap

Even after the control flow fix, a consistent shadow → Verilator ratio of ~0.84 was observed:

| Shadow % | Verilator % | Ratio |
|---------|------------|-------|
| 30.20% | 25.75% | 0.853 |
| 34.17% | 28.83% | 0.844 |
| 36.42% | 30.68% | 0.842 |

This gap is attributed to remaining differences in register value computation for OPERAND_PAIR bins and the exclusion of the prologue from shadow coverage tracking.

---

## 8. Next Steps

- Current training: 2000 episodes x 480 steps (same as constrained random baseline)
- RAW-aware observation space to guide policy toward harder-to-hit bins
- Validation: 300 episodes x 480 steps → direct comparison with constrained random
