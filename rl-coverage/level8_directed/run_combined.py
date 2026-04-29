"""run_combined.py — Rulează directed_l8 + directed_l8_targeted într-un singur run.

Un singur run Verilator = un singur coverage.dat cumulat.
"""

import json, os, sys, subprocess
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
import cov_parser

ML4DV  = (THIS.parent.parent / "cpu").resolve()
_SIM_BUILD = os.environ.get("IBEX_SIM_BUILD", "sim_build")
VTOP   = ML4DV / _SIM_BUILD / "Vtop"
COVDAT = ML4DV / "coverage.dat"
PROGRAM_JSON = "/tmp/rl_l8_combined.json"

from directed_l8 import build_directed_program
from directed_l8_targeted import build_targeted_program

def run_combined():
    prog_base     = build_directed_program()
    prog_targeted = build_targeted_program()
    prog = prog_base + prog_targeted

    print(f"Program combinat: {len(prog_base)} + {len(prog_targeted)} = {len(prog)} instrucțiuni")

    payload = {"n": len(prog), "agent": "combined",
               "machine_code": [int(w) for w in prog]}
    with open(PROGRAM_JSON, "w") as f:
        json.dump(payload, f)

    import sysconfig as _sc
    _pylib = _sc.get_config_var("LIBDIR") or ""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu"
        + ((":" + _pylib) if _pylib else "")
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = PROGRAM_JSON

    print(f"Running Vtop...")
    r = subprocess.run([str(VTOP)], cwd=str(ML4DV), env=env,
                       capture_output=True, text=True, timeout=600)
    print("[OK]" if r.returncode == 0 else f"[WARN] rc={r.returncode}\n{r.stderr[-500:]}")

    s = cov_parser.parse(str(COVDAT))
    for k in ("toggle", "branch", "line"):
        c, t = s.by_kind[k]
        print(f"  {k:7s}: {c:>5}/{t:>5}  ({100*c/t:.2f}%)")

    print("\nTop-10 uncovered (line):")
    rows = sorted([(t-c, pg.replace("v_line/",""), c, t)
                   for pg,(c,t) in s.by_page.items()
                   if pg.startswith("v_line/")], reverse=True)
    for miss, mod, c, t in rows[:10]:
        print(f"  {miss:>4} miss  {c:>4}/{t:<4}  {100*c/t:.1f}%  {mod}")

if __name__ == "__main__":
    run_combined()
