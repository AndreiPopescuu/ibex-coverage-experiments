"""env_l9_shadow.py — Gym environment cu Python shadow (fără Verilator).

Avantaje față de env_l9.py:
  • ~1000× mai rapid (nu pornește subprocess Verilator)
  • Observation space bogat — 14 features
  • 5000+ episoade fezabile pe același buget de timp

Obs space (14 features, toate [0,1]):
  [0]  step_frac            — poziție în episod
  [1]  cum_cov_frac         — coverage cumulat global / total_bins
  [2]  seen_frac            — bins SEEN acoperite cumulat
  [3]  opair_frac           — bins OPERAND_PAIR acoperite cumulat
  [4]  raw_frac             — bins RAW_HAZARD acoperite cumulat
  [5]  seq_frac             — bins SEQUENCE acoperite cumulat
  [6]  corner_frac          — bins CORNER acoperite cumulat
  [7]  ep_new_frac          — bins noi în episodul curent / total_bins
  [8]  last_op_norm         — ultimul opcode normalizat
  [9]  ep_idx_norm          — nr. episod normalizat
  [10] last_rd_norm         — ultimul rd normalizat
  [11] recent_rds_frac      — câți producători recenti / 3
  [12] seq_covered_last_op  — fracție seq bins ale ultimei instrucțiuni deja acoperite
  [13] raw_uncov_consumer   — fracție RAW bins neacoperite unde ultima instr e consumer

Reward end-of-episode:
  +10 × bins_noi
  -1  dacă 0 bins noi
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level8_directed"))

from shadow_riscv import RISCVShadow
from coverage_model import (CoverageModel, N_INSTRS, INSTR_NAMES,
                              HAS_RD, HAS_RS1, HAS_RS2, HAS_IMM, N_OP_CATS)
from codec_l8 import N_OPS, IMM_BUCKETS, encode


# ── Precompute total bins per categorie ───────────────────────────────────────

def _compute_totals():
    seen_t   = N_INSTRS
    op_t     = sum(N_OP_CATS * N_OP_CATS for n in INSTR_NAMES if n in HAS_RS1 or n in HAS_RS2)
    raw_t    = 3 * sum(1 for pn in INSTR_NAMES if pn in HAS_RD
                         for cn in INSTR_NAMES if cn in HAS_RS1 or cn in HAS_RS2)
    seq_t    = N_INSTRS * N_INSTRS
    corner_t = len(CoverageModel()._make_corner_bins())
    rdzero_t = len(HAS_RD)
    rs1zero_t= len(HAS_RS1)
    imm_t    = len(HAS_IMM) * N_OP_CATS
    samesrc_t= len(HAS_RS2)
    _, real_total = CoverageModel().count_bins()
    return seen_t, op_t, raw_t, seq_t, corner_t, rdzero_t, rs1zero_t, imm_t, samesrc_t, real_total


(_SEEN_T, _OPAIR_T, _RAW_T, _SEQ_T, _CORNER_T,
 _RDZERO_T, _RS1ZERO_T, _IMM_T, _SAMESRC_T, _TOTAL_BINS) = _compute_totals()

# ── Precompute lookups bin_name → tuple (pentru update rapid la end-of-episode) ──

_SEQ_NAME_TO_PAIR: dict = {
    f"seq_{pn}_{cn}": (i, j)
    for i, pn in enumerate(INSTR_NAMES)
    for j, cn in enumerate(INSTR_NAMES)
}

_RAW_NAME_TO_TRIPLE: dict = {
    f"raw{d+1}_{pn}_{cn}": (d, i, j)   # d=0,1,2 corespunde distanțelor 1,2,3
    for d in range(3)
    for i, pn in enumerate(INSTR_NAMES)
    for j, cn in enumerate(INSTR_NAMES)
}

# Numărul de producători valizi (au RD) — pentru normalizare obs RAW consumer
_N_PRODUCERS = sum(1 for n in INSTR_NAMES if n in HAS_RD)
_RAW_CONSUMER_TOTAL = max(3 * _N_PRODUCERS, 1)   # per consumer instruction


class IbexL9ShadowEnv(gym.Env):
    """Environment rapid pentru L9 folosind shadow RISC-V ISA simulator."""

    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 256, max_episodes: int = 5000):
        super().__init__()
        self.episode_steps = episode_steps
        self.max_episodes  = max_episodes

        self.action_space = spaces.MultiDiscrete([N_OPS, 32, 32, 32, IMM_BUCKETS])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )

        self._shadow     = RISCVShadow()
        self._ep_model   = CoverageModel()
        self._cum_hits   : set = set()
        self._ep_new     : set = set()

        self._step_idx   = 0
        self._n_episodes = 0
        self._last_op    = 0
        self._last_rd    = 0
        self._recent_rds : list = []   # ultimii 3 rd-uri (indice registru)
        self._recent_ops : list = []   # op_i corespunzător fiecărui rd recent
        self._actions    : list = []

        # Cache contoare per-categorie
        self._cat = {"seen": 0, "opair": 0, "raw": 0, "seq": 0, "corner": 0,
                     "rdzero": 0, "rs1zero": 0, "imm": 0, "samesrc": 0}

        # Seturi cumulate de perechi SEQ și RAW (tuple-uri de int, mai rapide decât string)
        self._cum_seq_pairs : set = set()   # (prev_op_i, curr_op_i)
        self._cum_raw_pairs : set = set()   # (dist_idx, prod_op_i, cons_op_i)  dist_idx ∈ {0,1,2}

        # Deduplicare în cadrul episodului curent
        self._ep_seq_new : set = set()
        self._ep_raw_new : set = set()

    # ── Helpers obs ──────────────────────────────────────────────────────────

    def _seq_covered_frac_for_op(self, op_i: int) -> float:
        """Fracția perechilor seq_{op_i}_* deja acoperite cumulat."""
        if op_i >= N_INSTRS or _SEQ_T == 0:
            return 0.0
        covered = sum(1 for j in range(N_INSTRS) if (op_i, j) in self._cum_seq_pairs)
        return covered / N_INSTRS

    def _raw_uncov_consumer_frac_for_op(self, op_i: int) -> float:
        """Fracția RAW bins neacoperite unde op_i e consumer."""
        if op_i >= N_INSTRS or INSTR_NAMES[op_i] not in HAS_RS1 and INSTR_NAMES[op_i] not in HAS_RS2:
            return 0.0
        covered = sum(
            1 for d in range(3) for i in range(N_INSTRS)
            if INSTR_NAMES[i] in HAS_RD and (d, i, op_i) in self._cum_raw_pairs
        )
        return 1.0 - min(1.0, covered / _RAW_CONSUMER_TOTAL)

    def _raw_potential(self, rs1: int, rs2: int) -> float:
        return 1.0 if any(
            rd != 0 and (rs1 == rd or rs2 == rd) for rd in self._recent_rds
        ) else 0.0

    def _obs(self) -> np.ndarray:
        op_i = self._last_op
        return np.array([
            self._step_idx / max(self.episode_steps, 1),              # [0]
            min(1.0, len(self._cum_hits) / _TOTAL_BINS),              # [1]
            min(1.0, self._cat["seen"]   / max(_SEEN_T,    1)),       # [2]
            min(1.0, self._cat["opair"]  / max(_OPAIR_T,   1)),       # [3]
            min(1.0, self._cat["raw"]    / max(_RAW_T,     1)),       # [4]
            min(1.0, self._cat["seq"]    / max(_SEQ_T,     1)),       # [5]
            min(1.0, self._cat["corner"] / max(_CORNER_T,  1)),       # [6]
            min(1.0, len(self._ep_new) / _TOTAL_BINS),                # [7]
            self._last_op / max(N_OPS - 1, 1),                        # [8]
            min(1.0, self._n_episodes / max(self.max_episodes, 1)),   # [9]
            self._last_rd / 31.0,                                     # [10]
            float(len(self._recent_rds)) / 3.0,                       # [11]
            self._seq_covered_frac_for_op(op_i),                      # [12]
            self._raw_uncov_consumer_frac_for_op(op_i),               # [13]
            min(1.0, self._cat["rdzero"]  / max(_RDZERO_T,  1)),      # [14]
            min(1.0, self._cat["rs1zero"] / max(_RS1ZERO_T, 1)),      # [15]
            min(1.0, self._cat["imm"]     / max(_IMM_T,     1)),      # [16]
            min(1.0, self._cat["samesrc"] / max(_SAMESRC_T, 1)),      # [17]
        ], dtype=np.float32)

    def _update_cat_cache(self, new_hits: set):
        for h in new_hits:
            if   h.startswith("seen_"):    self._cat["seen"]    += 1
            elif h.startswith("opair_"):   self._cat["opair"]   += 1
            elif h.startswith("raw"):      self._cat["raw"]     += 1
            elif h.startswith("seq_"):     self._cat["seq"]     += 1
            elif h.startswith("corner_"):  self._cat["corner"]  += 1
            elif h.startswith("rdzero_"):  self._cat["rdzero"]  += 1
            elif h.startswith("rs1zero_"): self._cat["rs1zero"] += 1
            elif h.startswith("imm_"):     self._cat["imm"]     += 1
            elif h.startswith("samesrc_"): self._cat["samesrc"] += 1

    def _update_pair_caches(self, new_hits: set):
        """Actualizează _cum_seq_pairs și _cum_raw_pairs din bin-urile noi."""
        for h in new_hits:
            if h.startswith("seq_"):
                pair = _SEQ_NAME_TO_PAIR.get(h)
                if pair:
                    self._cum_seq_pairs.add(pair)
            elif h.startswith("raw"):
                triple = _RAW_NAME_TO_TRIPLE.get(h)
                if triple:
                    self._cum_raw_pairs.add(triple)

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        self._shadow.reset()
        self._ep_model = CoverageModel()
        self._ep_new   = set()
        self._step_idx = 0
        self._last_op  = 0
        self._last_rd  = 0
        self._recent_rds.clear()
        self._recent_ops.clear()
        self._actions.clear()
        self._ep_seq_new.clear()
        self._ep_raw_new.clear()
        return self._obs(), {}

    def step(self, action):
        op_i, rd, rs1, rs2, imm_b = (int(x) for x in action)
        self._actions.append((op_i, rd, rs1, rs2, imm_b))
        self._step_idx += 1

        reward = 0.0

        # ── Actualizare stare pentru obs ─────────────────────────────────────
        if rd != 0:
            self._recent_rds.append(rd)
            self._recent_ops.append(op_i)
            if len(self._recent_rds) > 3:
                self._recent_rds.pop(0)
                self._recent_ops.pop(0)
        self._last_op = op_i
        self._last_rd = rd

        truncated = self._step_idx >= self.episode_steps
        info      = {}

        if truncated:
            self._shadow.reset()
            result   = self._shadow.run_actions(self._actions)
            ep_hits  = set(result["hits"])
            new_hits = ep_hits - self._cum_hits
            self._ep_new = new_hits
            self._update_cat_cache(new_hits)
            self._update_pair_caches(new_hits)
            self._cum_hits |= ep_hits

            reward += float(len(new_hits)) * 10.0
            if len(new_hits) == 0:
                reward += -1.0

            cum_pct = 100.0 * len(self._cum_hits) / _TOTAL_BINS
            info.update({
                "ep_pct"      : result["pct"],
                "cum_pct"     : cum_pct,
                "new_hits"    : len(new_hits),
                "cum_covered" : len(self._cum_hits),
                "total_bins"  : _TOTAL_BINS,
            })
            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info
