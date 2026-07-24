#!/bin/bash
# run_l11_steps_curriculum.sh — episode-length curriculum for train_l11_ppo.py:
# starts with short episodes (frequent terminal reward -> better credit
# assignment early on, since reward only arrives once per episode) and grows
# towards full-length episodes as training progresses, so bins that need a
# long instruction sequence (e.g. PMP setup -> access, or a loop counter that
# must reach a specific value) stay reachable throughout — a fixed-short
# schedule would plateau below what long episodes can find, no matter how
# long you train.
#
# Each stage resumes from the previous stage's checkpoint (weights AND
# accumulated hits — _ckpt_paths() in train_l11_ppo.py keys checkpoints only
# by --phase, not by --steps, so switching --steps across stages intentionally
# reuses the same checkpoint file). No --hits is ever passed: this trains from
# a 0% baseline by design, to keep the RL-vs-constrained-random comparison
# clean (RL gets no head start from the testlist-suite's discoveries).
#
# Usage:
#   bash rl-coverage/level11_maxconfig/run_l11_steps_curriculum.sh [extra train_l11_ppo.py args...]
#
# Examples:
#   bash run_l11_steps_curriculum.sh --n-envs 4
#   bash run_l11_steps_curriculum.sh --phase 1 --action-mask
#   STAGES="64 128 256 256" EPISODES_PER_STAGE=900 bash run_l11_steps_curriculum.sh
#
# Env overrides:
#   STAGES              default "64 128 256"   episode lengths, in order
#   EPISODES_PER_STAGE  default 1200           episodes trained at each length
#                                               (total episodes = this * len(STAGES))
#   IBEX_VENV           default $HOME/ibex_env
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${IBEX_VENV:-$HOME/ibex_env}/bin/activate"
cd "$SCRIPT_DIR"

read -ra STAGE_LIST <<< "${STAGES:-64 128 256}"
EPISODES_PER_STAGE="${EPISODES_PER_STAGE:-1200}"

echo "=== L11 episode-length curriculum: ${STAGE_LIST[*]} steps, "\
"${EPISODES_PER_STAGE} episodes/stage (total $((EPISODES_PER_STAGE * ${#STAGE_LIST[@]}))) ==="

cumulative=0
for steps in "${STAGE_LIST[@]}"; do
    cumulative=$((cumulative + EPISODES_PER_STAGE))
    echo ""
    echo "=== Stage: --steps $steps, episode target $cumulative ==="
    # --resume on a non-existent checkpoint just starts fresh (train_l11_ppo.py
    # prints a warning and continues) — so this is safe as the very first stage too,
    # and safe to re-run the whole script if a stage got interrupted midway.
    python3 train_l11_ppo.py --steps "$steps" --episodes "$cumulative" --resume "$@"
done
