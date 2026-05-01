"""
Decoder environment with RAW (unstructured) action space.

The agent chooses a 32-bit instruction directly as 4 bytes
(MultiDiscrete([256, 256, 256, 256])) with no knowledge of
op/rd/rs1/rs2 field structure. The env decodes it the same
way random_baseline_raw32 does.

This is the fair comparison with ML4DV's LLM approach, which
also generates raw hex instructions without a structured action space.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shadow_decoder import N_BINS, bins_for_action, OP_TYPES

_OP_KEY = {(k, n): i for i, (k, n) in enumerate(OP_TYPES)}

_ALU_F3 = {
    (0b000, 0): "add",  (0b000, 1): "sub",  (0b100, 0): "xor",
    (0b110, 0): "or",   (0b111, 0): "and",  (0b001, 0): "sll",
    (0b101, 0): "srl",  (0b101, 1): "sra",  (0b010, 0): "slt",
    (0b011, 0): "sltu",
}
_ALUI_F3 = {
    (0b000, 0): "add",  (0b100, 0): "xor",  (0b110, 0): "or",
    (0b111, 0): "and",  (0b001, 0): "sll",  (0b101, 0): "srl",
    (0b101, 1): "sra",  (0b010, 0): "slt",  (0b011, 0): "sltu",
}
_LOAD_F3  = {0b010: "word", 0b001: "half-word", 0b000: "byte"}
_STORE_F3 = {0b010: "word", 0b001: "half-word", 0b000: "byte"}


def decode_raw(word: int):
    """Decode a 32-bit integer to (op_idx, rd, rs1, rs2) or None."""
    opcode = word & 0x7F
    rd     = (word >> 7)  & 0x1F
    rs1    = (word >> 15) & 0x1F
    rs2    = (word >> 20) & 0x1F
    f3     = (word >> 12) & 0x7
    f7b    = (word >> 30) & 0x1

    if opcode == 0b0110011:
        name = _ALU_F3.get((f3, f7b))
        if name: return (_OP_KEY[("alu", name)], rd, rs1, rs2)
    elif opcode == 0b0010011:
        name = _ALUI_F3.get((f3, f7b))
        if name: return (_OP_KEY[("alu_imm", name)], rd, rs1, rs2)
        # SUBI: opcode correct, funct3=0, f7b=0 but sub doesn't exist as I-type
        if f3 == 0b000:
            return (_OP_KEY[("alu_imm", "sub")], rd, rs1, rs2)
    elif opcode == 0b0000011:
        sz = _LOAD_F3.get(f3)
        if sz: return (_OP_KEY[("load", sz)], rd, rs1, rs2)
    elif opcode == 0b0100011:
        sz = _STORE_F3.get(f3)
        if sz: return (_OP_KEY[("store", sz)], rd, rs1, rs2)
    return None


class IbexDecoderRawEnv(gym.Env):
    """
    Action: MultiDiscrete([256, 256, 256, 256]) = 4 raw bytes of instruction.
    No field structure given to the agent — it must discover op/rd/rs1/rs2 itself.
    """
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 256, seed: int | None = None):
        super().__init__()
        self.episode_steps = episode_steps
        self.action_space = spaces.MultiDiscrete([256, 256, 256, 256])
        self.observation_space = spaces.MultiBinary(N_BINS)
        self._rng = np.random.default_rng(seed)
        self.covered = np.zeros(N_BINS, dtype=np.int8)
        self.step_idx = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.covered[:] = 0
        self.step_idx = 0
        return self.covered.copy(), {}

    def step(self, action):
        b0, b1, b2, b3 = [int(x) for x in action]
        word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        new_hits = 0
        decoded = decode_raw(word)
        if decoded is not None:
            op_idx, rd, rs1, rs2 = decoded
            for b in bins_for_action(op_idx, rd, rs1, rs2):
                if not self.covered[b]:
                    self.covered[b] = 1
                    new_hits += 1
        self.step_idx += 1
        terminated = bool(self.covered.sum() == N_BINS)
        truncated  = self.step_idx >= self.episode_steps
        return self.covered.copy(), float(new_hits), terminated, truncated, {
            "covered": int(self.covered.sum()),
        }
