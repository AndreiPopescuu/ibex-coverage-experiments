"""env_l11.py — env for the "max" Ibex build (cocotb_ibex_max.sv: PMP,
ICache, RV32B, lockstep, ... all on), using codec_l11 (143 ops:
L10's 87 RV32IMC ops + 29 RV32B Zba/Zbb/Zbs + 27 RV32B zbp/zbc/zbe/zbf).

Build the matching Vtop once with:
    cd cpu && make CONFIG=max

This does NOT touch cpu/sim_build/ or cpu/coverage.dat (the minimal-config
build) — output goes to cpu/sim_build_max/ and cpu/coverage_max.dat.
"""

import os, sys, re, json, subprocess, sysconfig
from collections import Counter, deque
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

def _cocotb_libpython_dir() -> str | None:
    """Dir containing libpythonX.Y.so that cocotb's gpi was built against —
    matches cpu/run_testlist_suite.sh's `cocotb-config --libpython` lookup.
    Needed because guessing /usr/lib/x86_64-linux-gnu (the old fallback here)
    silently breaks on any non-system-Python venv (e.g. a venv layered on
    Anaconda/Miniconda, whose libpython lives under the conda install dir):
    gpi fails to dlopen it, so the embedded Python side of the test
    (test_run_for_l8.py, which actually injects the RL program) never runs —
    but Verilator still reaches $finish and dumps a "no test ran" coverage
    snapshot with returncode 0, silently identical on every call regardless
    of the program fed in.
    """
    try:
        libpython = subprocess.check_output(
            ["cocotb-config", "--libpython"], text=True).strip()
        return os.path.dirname(libpython)
    except Exception:
        return None

_COCOTB_LIBPYTHON_DIR = _cocotb_libpython_dir()

THIS   = Path(__file__).resolve().parent
L5     = (THIS.parent / "level5_real_rtl").resolve()
L10    = (THIS.parent / "level10_ops").resolve()
ML4DV  = (THIS.parent.parent / "cpu").resolve()

# Two interchangeable Vtop builds, same test_run_for_l8 harness + codec_l11
# encoding, different RTL/module hierarchy:
#   "max"                -> sim_build_max/Vtop: old vendored tree, custom
#                           PMP/ICache/RV32B/lockstep config, 33624 toggle bins
#   "opentitan_upstream"  -> sim_build_opentitan_upstream/Vtop: fresh-vendored
#                           lowRISC/ibex master, official "opentitan" preset —
#                           the exact build the 171-test testlist-suite baseline
#                           (83.32% toggle) ran on, 38696 toggle bins.
VTOP_BUILDS = {
    "max":                (ML4DV / "sim_build_max" / "Vtop", "coverage_max"),
    "opentitan_upstream":  (ML4DV / "sim_build_opentitan_upstream" / "Vtop",
                             "coverage_opentitan_upstream"),
}
DEFAULT_BUILD = "max"
VTOP   = VTOP_BUILDS[DEFAULT_BUILD][0]

sys.path.insert(0, str(L5))
sys.path.insert(0, str(L10))
import cov_parser

from codec_l11 import N_OPS, IMM_BUCKETS, N_CSR_BUCKETS, L11_CSRS, emit_program

# Reused directly (not re-derived) so the RL agent's rd/rs1/rs2/imm_bucket
# selection is byte-identical to the testlist-suite's own constrained-random
# generator — see the override in IbexL11Env.step() below.
from constrained_random_l11 import _pick_reg, _pick_bucket, P_HAZARD, P_ZERO, P_SAME_SRC

# Curriculum phases: max op index (action[0] in [0, max_ops))
# Phase 1 (45 ops): core RV32I ALU, loads/stores, CSR, RV32M, branches, JAL
# Phase 2 (87 ops): + RV32C compressed, LUI/AUIPC/JALR/ECALL/EBREAK/FENCE,
#                     L9 extras (MRET/WFI/FENCE_I/C_J*/C_B*/...), L10 misalign
# Phase 3 (143 ops): full L11 — RV32B Zba/Zbb/Zbs (87->116) +
#                     zbp/zbc/zbe/zbf legacy draft ops (116->143)
PHASE_MAX_OPS: dict[int, int] = {1: 45, 2: 87, 3: N_OPS}

MODULES = [
    "ibex_core", "ibex_cs_registers", "ibex_top", "ibex_if_stage",
    "ibex_top_tracing", "ibex_alu", "ibex_id_stage", "ibex_multdiv_fast",
    "ibex_ex_block", "ibex_tracer",
    # ibex_prefetch_buffer absent: cu ICache=1 e înlocuit de gen_icache,
    # gen_prefetch_buffer nu se elaborează → zero bins în coverage_max.dat.
    "ibex_controller", "ibex_compressed_decoder", "ibex_register_file_ff",
    "ibex_load_store_unit", "ibex_decoder",
    # ibex_fetch_fifo absent: instanțiat în ibex_prefetch_buffer,
    # care nu se elaborează cu ICache=1 → 0 bins în coverage_max.dat.
    "ibex_counter", "ibex_csr", "ibex_wb_stage",
    "ibex_icache", "ibex_pmp", "ibex_lockstep",
    "ibex_dummy_instr",    # SecureIbex=1 → DummyInstructions=1
    "ibex_branch_predict", # BranchPredictor=1
    "cocotb_ibex", "prim_generic_clock_gating",
    "prim_clock_gating", "prim_buf", "prim_generic_buf",
]
N_MODULES   = len(MODULES)
HIST_LEN    = 4
N_OBS       = 3 + N_MODULES + HIST_LEN
MAX_EP_NORM = 500.0
BRANCH_COEF = 0.3
DYNAMIC_WEIGHT_CAP = 5.0

# ── Static op -> RTL-module mapping, used only for action masking (MaskablePPO,
# --action-mask in train_l11_ppo.py). This is a coarse, hand-built heuristic
# from the RISC-V opcode groups in codec_l10/codec_l11 (ADD/SUB/... -> ALU,
# LB/SW/... -> load-store, CSRRW/... -> CS registers, etc.) — NOT a claim about
# exact toggle-bin attribution, which we can't get per-instruction (coverage is
# only readable once per whole episode, see run_program()). Cross-cutting
# modules (decoder, controller, id/ex/wb stage, register file, lockstep,
# tracer, prim_*) are deliberately left unmapped: virtually every op touches
# them, so no op choice could selectively avoid them anyway.
_ALU_OPS = (
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     45, 46, 47, 48, 49, 50, 51, 54, 55, 56, 57, 58, 59, 60, 61, 64, 77}
    | set(range(87, N_OPS))  # all RV32B Zba/Zbb/Zbs/zbp/zbc/zbe/zbf route through the ALU
)
_MEM_OPS    = {19, 20, 21, 22, 23, 24, 25, 26, 52, 53, 78, 79, 84, 85, 86}
_CSR_OPS    = {27, 28, 29, 66, 67, 68}
_MULDIV_OPS = {30, 31, 32, 33, 34, 35, 36, 37}
_BRANCH_OPS = {38, 39, 40, 41, 42, 43, 75, 76}
_RVC_OPS    = {45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
               73, 74, 75, 76, 77, 78, 79, 80, 81, 82}
_ICACHE_OPS = {72}  # FENCE_I


def _build_op_modules() -> dict[int, tuple[str, ...]]:
    out: dict[int, tuple[str, ...]] = {}
    for op in range(N_OPS):
        mods: set[str] = set()
        if op in _ALU_OPS:
            mods.add("ibex_alu")
        if op in _MEM_OPS:
            mods.update(("ibex_load_store_unit", "ibex_pmp"))
        if op in _CSR_OPS:
            # CSR writes are the only path to pmpcfg/pmpaddr, to
            # mcycle/minstret/mhpmcounter* (ibex_counter), and to
            # cpuctrl/secureseed (dummy_instr config) -> tag all five; an op
            # only gets masked once ALL of its tagged modules saturate, so
            # CSR ops stay available as long as any one of these lags.
            mods.update(("ibex_cs_registers", "ibex_csr",
                         "ibex_pmp", "ibex_dummy_instr", "ibex_counter"))
        if op in _MULDIV_OPS:
            mods.add("ibex_multdiv_fast")
        if op in _BRANCH_OPS:
            mods.add("ibex_branch_predict")
        if op in _RVC_OPS:
            mods.add("ibex_compressed_decoder")
        if op in _ICACHE_OPS:
            mods.add("ibex_icache")
        out[op] = tuple(mods)
    return out


OP_MODULES: dict[int, tuple[str, ...]] = _build_op_modules()

# ── csr_bucket -> RTL-module map, used to extend action masking onto the
# csr_bucket sub-space (see action_masks() below) — the RL analogue of
# testlist_l11.py's _build_pmp_focus/_build_csr_sweep domain knowledge: those
# functions bias/sweep CSR *addresses* toward pmpcfg/pmpaddr and toward every
# implemented CSR respectively, because that's exactly what's needed to reach
# ibex_pmp's comparator toggles and ibex_counter's HPM-counter toggles. Rather
# than importing that CRT-specific file into the RL env, the same domain
# knowledge (which CSR addresses belong to which module) is re-derived here
# directly from address sets, sourced from codec_l11.py/codec_l7.py's L11_CSRS
# provenance comments. ────────────────────────────────────────────────────────
_PMP_ADDRS = {0x3A0, 0x3A1, 0x3A2, 0x3A3} | {0x3B0 + i for i in range(16)}
_COUNTER_ADDRS = (
    {0xB00, 0xB02, 0xB80, 0xB82, 0x320}          # mcycle/minstret[h]/mcountinhibit
    | {0xB03 + i for i in range(6)}              # mhpmcounter3..8
    | {0x323 + i for i in range(6)}              # mhpmevent3..8
    | {0xC00, 0xC01, 0xC02}                      # cycle/time/instret (user shadow)
    | {0xB09 + i for i in range(6)}              # mhpmcounter9..14
    | {0x329 + i for i in range(6)}              # mhpmevent9..14
)
_DUMMY_INSTR_ADDRS = {0x7C0, 0x7C1}  # cpuctrl, secureseed


def _build_csr_bucket_modules() -> dict[int, str]:
    out: dict[int, str] = {}
    for i, addr in enumerate(L11_CSRS):
        if addr in _PMP_ADDRS:
            out[i] = "ibex_pmp"
        elif addr in _COUNTER_ADDRS:
            out[i] = "ibex_counter"
        elif addr in _DUMMY_INSTR_ADDRS:
            out[i] = "ibex_dummy_instr"
        else:
            out[i] = "ibex_cs_registers"
    return out


CSR_BUCKET_MODULES: dict[int, str] = _build_csr_bucket_modules()
assert len(CSR_BUCKET_MODULES) == N_CSR_BUCKETS
assert sum(1 for m in CSR_BUCKET_MODULES.values() if m == "ibex_pmp") == 20
assert sum(1 for m in CSR_BUCKET_MODULES.values() if m == "ibex_dummy_instr") == 2
assert sum(1 for m in CSR_BUCKET_MODULES.values() if m == "ibex_counter") == \
    len(_COUNTER_ADDRS), "L11_CSRS is missing a counter CSR address"

MASK_SATURATION_DEFAULT = 0.97
MASK_MIN_UNMASKED_FRAC  = 0.15

_F_RE     = re.compile(r"\x01f\x02[^\x01]+")
_N_RE     = re.compile(r"\x01n\x02[^\x01]+")
_H_DOT_RE = re.compile(r"(\x01h\x02)\.")

def _norm_key(key: str) -> str:
    """Cheie stabilă între mașini și build-uri: scoate f, n, dot din h."""
    k = _F_RE.sub("", key)
    k = _N_RE.sub("", k)
    k = _H_DOT_RE.sub(r"\1", k)
    return k


def _module_of(key: str) -> str | None:
    m = re.search(r"page\x02v_toggle/([^\x01]+)\x01", key)
    if not m:
        return None
    return m.group(1).split("__")[0]


def run_program(actions, program_json: str, covdat: Path, timeout: int = 600,
                 vtop: Path = VTOP):
    machine = emit_program(actions)
    with open(program_json, "w") as f:
        json.dump({"n": len(machine), "agent": "l11",
                   "machine_code": [int(m) for m in machine]}, f)
    env = os.environ.copy()
    site_packages = sysconfig.get_paths()["purelib"]
    cocotb_libs = os.path.join(site_packages, "cocotb", "libs")
    env["LD_LIBRARY_PATH"] = (
        (_COCOTB_LIBPYTHON_DIR + ":" if _COCOTB_LIBPYTHON_DIR else "")
        + cocotb_libs + ":/usr/lib/x86_64-linux-gnu"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = (
        str(ML4DV) + ":" + site_packages
        + ":" + env.get("PYTHONPATH", "")
    )
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = program_json
    proc = subprocess.run(
        [str(vtop), f"+verilator+coverage+file+{covdat}"],
        cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout,
    )
    if proc.returncode != 0:
        return None
    return cov_parser.parse(str(covdat))


class IbexL11Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 256, seed: int | None = None,
                 initial_hits: set | None = None, timeout: int = 600,
                 max_ops: int = N_OPS,
                 mask_saturation: float = MASK_SATURATION_DEFAULT,
                 mask_min_unmasked_frac: float = MASK_MIN_UNMASKED_FRAC,
                 build: str = DEFAULT_BUILD):
        super().__init__()
        if build not in VTOP_BUILDS:
            raise ValueError(f"Unknown build {build!r}, expected one of {list(VTOP_BUILDS)}")
        self._vtop, covdat_prefix = VTOP_BUILDS[build]
        if not self._vtop.exists():
            raise FileNotFoundError(
                f"Prebuilt Vtop for build {build!r} not found at {self._vtop}."
            )
        self.episode_steps = episode_steps
        self._timeout = timeout
        self._max_ops = max(1, min(max_ops, N_OPS))
        self._mask_saturation   = mask_saturation
        self._mask_min_unmasked = max(1, int(mask_min_unmasked_frac * self._max_ops))
        self._mask_min_unmasked_csr = max(1, int(mask_min_unmasked_frac * N_CSR_BUCKETS))

        # RNG + last-rd tracking for the CRT-identical rd/rs1/rs2/imm_bucket
        # override in step() below — separate from PPO's own action sampling,
        # mirrors constrained_random_l11.build_actions()'s last_rd threading.
        self._rng = np.random.default_rng(seed)
        self._last_rd: int | None = None

        # Path-uri unice per proces — esential pentru rulare paralela (SubprocVecEnv):
        # fara asta, mai multe instante ar scrie in acelasi /tmp/rl_l11.json si
        # acelasi coverage_max.dat si s-ar calca reciproc. os.getpid() e evaluat
        # aici (in __init__), deci reflecta corect PID-ul procesului worker, nu
        # PID-ul parintelui de la momentul import-ului modulului.
        pid = os.getpid()
        self._program_json = f"/tmp/rl_l11_{pid}.json"
        self._covdat        = ML4DV / f"{covdat_prefix}_{pid}.dat"

        self.action_space = spaces.MultiDiscrete(
            [self._max_ops, 32, 32, 32, IMM_BUCKETS, N_CSR_BUCKETS]
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS,), dtype=np.float32)

        self._actions:    list[tuple] = []
        self._step_idx:   int = 0
        self._n_episodes: int = 0
        self._total_tog:  int = 1

        self._cum_hits:        set = {_norm_key(k) for k in initial_hits} if initial_hits else set()
        self._cum_branch_hits: set = set()
        self._key_to_mod: dict = {}

        self._mod_covered   = {m: 0 for m in MODULES}
        self._mod_total     = {m: 1 for m in MODULES}
        self._mod_total_set = False

        self._action_hist: deque = deque([0] * HIST_LEN, maxlen=HIST_LEN)

        if self._cum_hits:
            for key in self._cum_hits:
                mod = _module_of(key)
                self._key_to_mod[key] = mod
                if mod and mod in self._mod_covered:
                    self._mod_covered[mod] += 1

    def _dynamic_weight(self, mod: str | None) -> float:
        # Capped at DYNAMIC_WEIGHT_CAP (not the raw 1/max(frac, 0.01), which
        # allows up to 100x per hit): early episodes on a near-empty module
        # would otherwise produce reward spikes in the thousands, and since
        # PPO's value target scales with that, a single outlier episode can
        # dominate several epochs of gradient updates.
        if mod and mod in self._mod_covered and self._mod_total.get(mod, 0) > 0:
            frac = self._mod_covered[mod] / self._mod_total[mod]
            return min(DYNAMIC_WEIGHT_CAP, 1.0 / max(frac, 0.01))
        return 1.0

    def action_masks(self) -> np.ndarray:
        """Flat bool mask for sb3-contrib's MaskablePPO, concatenated across
        all MultiDiscrete sub-spaces in order (op, rd, rs1, rs2, imm, csr) —
        see MaskableMultiCategoricalDistribution.apply_masking(). Two
        dimensions are actually masked, independently (sb3-contrib masks each
        MultiDiscrete sub-space simultaneously, not sequentially, so this
        isn't conditioned on "op is a CSR write" — that would need an
        autoregressive policy head):

        - `op`: disallowed once every module in OP_MODULES[op] is
          >= self._mask_saturation covered.
        - `csr_bucket`: disallowed once CSR_BUCKET_MODULES[bucket]'s single
          target module is >= self._mask_saturation covered. This is the RL
          analogue of the testlist-suite's CRT constraints (riscv_pmp_suite_test/
          riscv_csr_test biasing CSR *addresses* toward pmpcfg/pmpaddr and
          the full CSR sweep, see CSR_BUCKET_MODULES above): it restricts
          *which* CSR addresses remain choosable to the ones whose module
          still lags, without picking the address for the agent — the policy
          still freely chooses op/sequencing/timing and picks randomly (via
          its own learned distribution) among whatever csr_bucket values
          remain unmasked.

        rd/rs1/rs2/imm stay fully open.

        Guards against masking everything into a corner: both masks are
        computed from module totals that are only known after the first
        episode (self._mod_total_set), and each falls back to fully-open if
        fewer than its configured minimum would remain unmasked
        (self._mask_min_unmasked / self._mask_min_unmasked_csr).
        """
        op_mask  = np.ones(self._max_ops, dtype=bool)
        csr_mask = np.ones(N_CSR_BUCKETS, dtype=bool)
        if self._mod_total_set:
            for op in range(self._max_ops):
                mods = OP_MODULES.get(op, ())
                if not mods:
                    continue
                if all(self._mod_covered[m] / max(self._mod_total[m], 1)
                       >= self._mask_saturation for m in mods):
                    op_mask[op] = False
            if op_mask.sum() < self._mask_min_unmasked:
                op_mask[:] = True

            for bucket in range(N_CSR_BUCKETS):
                mod = CSR_BUCKET_MODULES.get(bucket)
                if mod and (self._mod_covered[mod] / max(self._mod_total[mod], 1)
                            >= self._mask_saturation):
                    csr_mask[bucket] = False
            if csr_mask.sum() < self._mask_min_unmasked_csr:
                csr_mask[:] = True

        mid_dims = int(sum(self.action_space.nvec[1:-1]))  # rd, rs1, rs2, imm
        return np.concatenate([op_mask, np.ones(mid_dims, dtype=bool), csr_mask])

    def _obs(self) -> np.ndarray:
        step_frac = self._step_idx / self.episode_steps
        cum_cov   = len(self._cum_hits) / max(self._total_tog, 1)
        n_ep_frac = min(1.0, self._n_episodes / MAX_EP_NORM)
        mod_fracs = [self._mod_covered[m] / max(self._mod_total[m], 1)
                     for m in MODULES]
        hist_norm = [h / N_OPS for h in self._action_hist]
        return np.array(
            [step_frac, cum_cov, n_ep_frac] + mod_fracs + hist_norm,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        self._actions.clear()
        self._step_idx = 0
        self._action_hist = deque([0] * HIST_LEN, maxlen=HIST_LEN)
        self._last_rd = None  # fresh program -> no "previous instruction" to hazard off of
        return self._obs(), {}

    def step(self, action):
        op = int(action[0])
        _, _, _, _, _, csr_bucket = (int(x) for x in action)

        # CRT-identical override: rd/rs1/rs2/imm_bucket are replaced with
        # constrained_random_l11.py's own sample_action() field logic
        # (same functions, same P_HAZARD/P_ZERO/P_SAME_SRC constants, same
        # boundary-biased bucket picker) instead of whatever the policy
        # network output for those dimensions — these are the "syntactic"
        # fields the testlist-suite hand-tunes to hit specific coverpoints
        # (RAW_HAZARD/ZERO_DST/ZERO_SRC/SAME_SRC), not ones RL gains anything
        # by learning itself. `op` (what to emit) and `csr_bucket` (already
        # mask-restricted to under-covered CSR modules, see action_masks())
        # stay under RL's own control — that's where sequencing/timing
        # decisions actually live.
        rd  = _pick_reg(self._rng, self._last_rd, prefer_hazard=False,
                         p_hazard=P_HAZARD, p_zero=P_ZERO)
        rs1 = _pick_reg(self._rng, self._last_rd, prefer_hazard=True,
                         p_hazard=P_HAZARD, p_zero=P_ZERO)
        if self._rng.random() < P_SAME_SRC:
            rs2 = rs1
        else:
            rs2 = _pick_reg(self._rng, self._last_rd, prefer_hazard=True,
                             p_hazard=P_HAZARD, p_zero=P_ZERO)
        imm_bucket = _pick_bucket(self._rng, IMM_BUCKETS)
        self._last_rd = rd

        self._action_hist.append(op)
        self._actions.append((op, rd, rs1, rs2, imm_bucket, csr_bucket))
        self._step_idx += 1
        truncated = self._step_idx >= self.episode_steps
        reward, info = 0.0, {}

        if truncated:
            summary = run_program(self._actions, self._program_json, self._covdat,
                                   timeout=self._timeout, vtop=self._vtop)
            if summary is None:
                info["vtop_failed"] = True
            else:
                toggle_covered, toggle_total = summary.by_kind["toggle"]
                self._total_tog = max(toggle_total, 1)

                tog_prefix    = "\x01page\x02v_toggle/"
                branch_prefix = "\x01page\x02v_branch/"

                ep_hits = {_norm_key(k) for k, v in summary.points.items()
                           if v > 0 and tog_prefix in ("\x01" + k)}
                ep_branch_hits = {_norm_key(k) for k, v in summary.points.items()
                                  if v > 0 and branch_prefix in ("\x01" + k)}

                new_hits        = ep_hits - self._cum_hits
                new_branch_hits = ep_branch_hits - self._cum_branch_hits

                self._cum_hits        |= ep_hits
                self._cum_branch_hits |= ep_branch_hits

                for key in ep_hits:
                    if key not in self._key_to_mod:
                        self._key_to_mod[key] = _module_of(key)

                if not self._mod_total_set:
                    self._mod_total_set = True
                    mod_counts = Counter()
                    for key in summary.points:
                        nk = _norm_key(key)
                        if nk not in self._key_to_mod:
                            self._key_to_mod[nk] = _module_of(key)
                        mod = self._key_to_mod[nk]
                        if mod:
                            mod_counts[mod] += 1
                    for mod in MODULES:
                        if mod in mod_counts:
                            self._mod_total[mod] = mod_counts[mod]

                for key in new_hits:
                    mod = self._key_to_mod.get(key)
                    if mod and mod in self._mod_covered:
                        self._mod_covered[mod] += 1

                shaped_toggle = sum(
                    self._dynamic_weight(self._key_to_mod.get(k))
                    for k in new_hits
                )
                branch_reward = BRANCH_COEF * len(new_branch_hits)
                reward = shaped_toggle + branch_reward

                cum_pct   = 100.0 * len(self._cum_hits) / self._total_tog
                # _mod_total[m] stays at its init value of 1 forever for any
                # module that never actually appears in the coverage.dat for
                # this build (e.g. ibex_branch_predict under opentitan_upstream,
                # where BranchPredictor=0 means it's never elaborated) -- such
                # a module is permanently stuck at 0/1=0.0% and would always
                # win min(), making "worst module" useless for diagnosing real
                # gaps. Excluded here (display/diagnostic only -- masking in
                # action_masks() is unaffected).
                _real_mods = [m for m in MODULES if self._mod_total[m] > 1]
                worst_mod = min(_real_mods or MODULES,
                                key=lambda m: self._mod_covered[m] / max(self._mod_total[m], 1))
                branch_total = max(summary.by_kind["branch"][1], 1)
                info.update({
                    "ep_pct":           100.0 * toggle_covered / self._total_tog,
                    "cum_covered":      len(self._cum_hits),
                    "cum_pct":          cum_pct,
                    "new_hits_vs_cum":  len(new_hits),
                    "new_branch_hits":  len(new_branch_hits),
                    "shaped_reward":    shaped_toggle,
                    "branch_reward":    branch_reward,
                    "branch_pct":       100.0 * summary.by_kind["branch"][0] / branch_total,
                    "worst_mod":        worst_mod,
                    "worst_pct":        100.0 * self._mod_covered[worst_mod] /
                                        max(self._mod_total[worst_mod], 1),
                    "mod_coverage":     {m: self._mod_covered[m] /
                                         max(self._mod_total[m], 1)
                                         for m in MODULES},
                    "ep_words":         [int(w) for w in emit_program(self._actions)]
                                        if new_hits else None,
                })
            self._n_episodes += 1

        return self._obs(), reward, False, truncated, info
