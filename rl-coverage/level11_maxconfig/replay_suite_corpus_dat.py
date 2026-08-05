"""replay_suite_corpus_dat.py — replay one corpus_suite_*.json's machine_code
through Verilator on the opentitan_upstream build, writing
cpu/coverage_suite_<name>.dat -- the exact format aggregate_suite_coverage.py
expects. Idempotent (skips if the target .dat already exists), so it's safe
to re-run over the whole corpus_suite_*.json set after an interruption.

Usage (one file):
    python3 replay_suite_corpus_dat.py corpus_suite_riscv_loop_test_seed0.json

Usage (all, parallel, EXCLUDING riscv_csr_test — see replay_all_suite_corpora.sh):
    ls corpus_suite_*.json | grep -v riscv_csr_test | xargs -P 8 -I{} \
        python3 replay_suite_corpus_dat.py {}
"""
import json, os, subprocess, sys, sysconfig
from pathlib import Path

THIS  = Path(__file__).resolve().parent
ML4DV = (THIS.parent.parent / "cpu").resolve()
VTOP  = ML4DV / "sim_build_opentitan_upstream" / "Vtop"

sys.path.insert(0, str(THIS.parent / "level5_real_rtl"))
import cov_parser  # noqa: E402


def _cocotb_libpython_dir():
    try:
        libpython = subprocess.check_output(
            ["cocotb-config", "--libpython"], text=True).strip()
        return os.path.dirname(libpython)
    except Exception:
        return None


def run_words(words, covdat: Path, tag: str) -> bool:
    program_json = f"/tmp/suite_replay_{tag}.json"
    with open(program_json, "w") as f:
        json.dump({"n": len(words), "agent": "suite_replay",
                   "machine_code": [int(w) for w in words]}, f)
    env = os.environ.copy()
    site_packages = sysconfig.get_paths()["purelib"]
    cocotb_libs = os.path.join(site_packages, "cocotb", "libs")
    libpython_dir = _cocotb_libpython_dir()
    env["LD_LIBRARY_PATH"] = (
        (libpython_dir + ":" if libpython_dir else "")
        + cocotb_libs + ":/usr/lib/x86_64-linux-gnu"
        + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = str(ML4DV) + ":" + site_packages + ":" + env.get("PYTHONPATH", "")
    env["MODULE"]     = "test_run_for_l8"
    env["RL_L8_JSON"] = program_json
    proc = subprocess.run(
        [str(VTOP), f"+verilator+coverage+file+{covdat}"],
        cwd=str(ML4DV), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600,
    )
    os.remove(program_json)
    return proc.returncode == 0


def main():
    if not VTOP.exists():
        sys.exit(f"Vtop not found at {VTOP} — build it first (cd cpu && "
                  f"make CONFIG=opentitan_upstream, or the project's equivalent).")

    corpus_path = Path(sys.argv[1])
    name = corpus_path.name[len("corpus_suite_"):-len(".json")]
    covdat = ML4DV / f"coverage_suite_{name}.dat"
    if covdat.exists():
        print(f"[skip] {name} (already have .dat)")
        return

    with open(corpus_path) as f:
        corpus = json.load(f)
    words = corpus["machine_code"]

    ok = run_words(words, covdat, name)
    if not ok:
        print(f"[FAIL] {name}")
        sys.exit(1)
    print(f"[ok] {name} ({len(words)} words)")


if __name__ == "__main__":
    main()
