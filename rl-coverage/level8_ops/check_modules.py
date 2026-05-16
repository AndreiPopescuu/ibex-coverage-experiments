"""check_modules.py — verifică ce module pot fi atinse.

Rulează un program simplu (NOP-uri + câteva instrucțiuni diverse),
parsează coverage.dat și afișează per modul:
  - total toggle points
  - câte sunt acoperite
  - dacă modulul e "mort" (total=0) sau "neatins" (covered=0)
"""

import sys
from pathlib import Path
from collections import defaultdict

THIS  = Path(__file__).resolve().parent
L5    = (THIS.parent / "level5_real_rtl").resolve()
sys.path.insert(0, str(L5))
import cov_parser

from directed_l8 import run_raw
from env_l8_v3 import MODULES

# Program cu instrucțiuni diverse ca să atingem cât mai mult din prima
PROGRAM = (
    [0x00000013] * 8    +  # NOP (ADDI x0, x0, 0)
    [0x00100093] * 4    +  # ADDI x1, x0, 1
    [0x00200113] * 4    +  # ADDI x2, x0, 2
    [0x002081b3]        +  # ADD x3, x1, x2
    [0x40208233]        +  # SUB x4, x1, x2
    [0x0020a2b3]        +  # SLT x5, x1, x2
    [0x02208033]        +  # MUL x0, x1, x2
    [0x0220c033]        +  # DIV x0, x1, x2
    [0x00102573]        +  # CSRRS x10, minstret, x0  (read minstret)
    [0x00002573]        +  # CSRRS x10, cycle, x0     (read cycle)
    [0x30002573]        +  # CSRRS x10, mstatus, x0
    [0x340015f3]        +  # CSRRW x11, mscratch, x0
    [0x00008067]        +  # JALR x0, 0(x1)
    [0x000010b7]        +  # LUI x1, 1
    [0x0FF0000F]        +  # FENCE iorw, iorw
    [0x00000073]        +  # ECALL
    [0x00100073]        +  # EBREAK
    [0x00000013] * 16      # NOP padding
)

def main():
    print("Rulează program de test... ", end="", flush=True)
    summary = run_raw(PROGRAM)
    if summary is None:
        print("EROARE: simulatorul a eșuat.")
        sys.exit(1)
    print("OK\n")

    # Coverage per pagină → per modul
    mod_data = defaultdict(lambda: {"covered": 0, "total": 0})
    for page, (c, t) in summary.by_page.items():
        if not page.startswith("v_toggle/"):
            continue
        mod = page[len("v_toggle/"):].split("__")[0]
        mod_data[mod]["covered"] += c
        mod_data[mod]["total"]   += t

    # Toate modulele unice din coverage
    all_mods = sorted(mod_data.keys())

    print(f"{'Modul':<35} {'Total':>7} {'Acoperit':>9} {'%':>7}  Status")
    print("-" * 72)

    tracked = set(MODULES)
    dead, untouched, partial, good = [], [], [], []

    for mod in all_mods:
        d = mod_data[mod]
        t, c = d["total"], d["covered"]
        pct = 100.0 * c / t if t > 0 else 0.0
        in_tracked = "(*)" if mod in tracked else "   "

        if t == 0:
            status = "MORT (total=0)"
            dead.append(mod)
        elif c == 0:
            status = "NEATINS (covered=0)"
            untouched.append(mod)
        elif pct < 30:
            status = f"scazut"
            partial.append(mod)
        else:
            status = "ok"
            good.append(mod)

        print(f"{mod:<35} {t:>7} {c:>9} {pct:>6.1f}%  {in_tracked} {status}")

    print("\n" + "=" * 72)
    print(f"Module moarte (total=0):       {dead}")
    print(f"Module neatinse (covered=0):   {untouched}")
    print(f"\n(*) = modul urmărit în MODULES din env_l8_v3.py")

    # Verifică dacă există module în MODULES care nu apar în coverage deloc
    missing = [m for m in MODULES if m not in mod_data]
    if missing:
        print(f"\nATENTIE: module din MODULES care lipsesc din coverage.dat:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
