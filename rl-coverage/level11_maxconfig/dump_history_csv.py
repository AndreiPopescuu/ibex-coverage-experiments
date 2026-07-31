"""dump_history_csv.py — export l11_*_checkpoint_history.npz (per-episode
ep/cum_pct/ep_pct, worker-local) to a plain CSV, so it can be committed and
read/plotted without needing numpy on the reading end.

Usage:
    python3 dump_history_csv.py [path/to/checkpoint_history.npz] [out.csv]

Defaults to l11_opentitan_upstream_checkpoint_history.npz -> rl_history.csv
"""
import sys
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent

npz_path = sys.argv[1] if len(sys.argv) > 1 else \
    str(THIS / "l11_opentitan_upstream_checkpoint_history.npz")
out_path = sys.argv[2] if len(sys.argv) > 2 else str(THIS / "rl_history.csv")

d = np.load(npz_path)
ep = d["ep"]
cum_pct = d["cum_pct"]
ep_pct = d["ep_pct"]

with open(out_path, "w") as f:
    f.write("ep,cum_pct,ep_pct\n")
    for i in range(len(ep)):
        f.write(f"{int(ep[i])},{cum_pct[i]:.4f},{ep_pct[i]:.4f}\n")

print(f"Wrote {len(ep)} rows -> {out_path}")
