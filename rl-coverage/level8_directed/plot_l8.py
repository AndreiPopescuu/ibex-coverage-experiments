"""Level 8 — Plot line/branch/toggle coverage curves.

Reads .npz files saved by greedy_l8.py and train_l8.py,
produces a comparison chart (l8_coverage.png).

Usage:
    python plot_l8.py                          # plots whatever .npz files exist
    python plot_l8.py --annotate               # also print per-module breakdown
"""

import argparse, sys
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — skipping chart, printing text summary only")


def load_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(str(path))
    return {k: d[k] for k in d.files}


def print_summary(label: str, data: dict):
    cum = data.get("cum_line_pct", np.array([]))
    ep  = data.get("ep_line_pct",  np.array([]))
    br  = data.get("branch_pct",   np.array([]))
    tog = data.get("toggle_pct",   np.array([]))
    if len(cum) == 0:
        return
    print(f"\n── {label} ──")
    print(f"  Episodes         : {len(cum)}")
    print(f"  Cum line (final) : {cum[-1]:.2f}%")
    print(f"  Cum line (peak)  : {cum.max():.2f}%")
    print(f"  Ep  line (mean)  : {ep.mean():.2f}%")
    print(f"  Ep  line (peak)  : {ep.max():.2f}%")
    if len(br):  print(f"  Branch (final)   : {br[-1]:.2f}%")
    if len(tog): print(f"  Toggle (final)   : {tog[-1]:.2f}%")


def annotate_coverage():
    """Print per-module line coverage from the current coverage.dat."""
    sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
    try:
        import cov_parser
        covdat = THIS.parent.parent / "cpu" / "coverage.dat"
        if not covdat.exists():
            print("[WARN] coverage.dat not found")
            return
        s = cov_parser.parse(str(covdat))
        print("\n── Per-module line coverage ──")
        rows = []
        for page, (c, t) in s.by_page.items():
            if page.startswith("v_line/"):
                rows.append((t - c, page.replace("v_line/", ""), c, t))
        rows.sort(reverse=True)
        for missed, mod, c, t in rows:
            bar = "█" * int(20 * c / max(t, 1))
            print(f"  {missed:>4} miss  {c:>4}/{t:<4} {100*c/t:5.1f}%  {bar:<20} {mod}")
        c_tot, t_tot = s.by_kind.get("line", (0, 1))
        print(f"\n  TOTAL line: {c_tot}/{t_tot} = {100*c_tot/t_tot:.2f}%")
    except Exception as e:
        print(f"[WARN] annotation failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", action="store_true")
    args = ap.parse_args()

    random_data  = load_npz(THIS / "l8_random_baseline.npz")
    ppo_data     = load_npz(THIS / "l8_ppo_curve.npz")

    # Text summaries
    if random_data: print_summary("Random baseline (L8)", random_data)
    if ppo_data:    print_summary("PPO (L8)",              ppo_data)

    if args.annotate:
        annotate_coverage()

    if not HAS_MPL:
        return
    if not random_data and not ppo_data:
        print("No .npz files found — run greedy_l8.py or train_l8.py first.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Level 8 — Line / Branch / Toggle Coverage", fontsize=13)

    colours = {"random": "#2196F3", "ppo": "#FF5722"}

    for ax, metric, ylabel in zip(
        axes,
        ["cum_line_pct", "branch_pct", "toggle_pct"],
        ["Cumulative Line Coverage (%)", "Branch Coverage (%)", "Toggle Coverage (%)"],
    ):
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if random_data and metric in random_data:
            y = random_data[metric]
            ax.plot(range(1, len(y)+1), y, color=colours["random"],
                    label=f"Random (final {y[-1]:.1f}%)", linewidth=1.5)

        if ppo_data and metric in ppo_data:
            y = ppo_data[metric]
            ax.plot(range(1, len(y)+1), y, color=colours["ppo"],
                    label=f"PPO (final {y[-1]:.1f}%)", linewidth=1.5)

        ax.legend(fontsize=9)

    plt.tight_layout()
    out = THIS / "l8_coverage.png"
    plt.savefig(str(out), dpi=150)
    print(f"\n[saved] {out}")
    plt.show()


if __name__ == "__main__":
    main()
