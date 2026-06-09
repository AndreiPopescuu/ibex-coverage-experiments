"""check_targets.py — verifica daca target bins din accessible_bins_for_llm.txt
sunt in baseline sau nu, si daca numele se potrivesc cu covdat_map.
"""
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

import starter

baseline_ids, covdat_map, target_bins = starter.setup()

# Inverseaza covdat_map: signal_name -> list of stable_ids
name_to_sids = {}
for sid, name in covdat_map.items():
    name_to_sids.setdefault(name, []).append(sid)

print(f"\nTotal target bins in accessible_bins_for_llm.txt: {len(target_bins)}")
print(f"Total entries in covdat_map: {len(covdat_map)}")
print(f"Baseline size: {len(baseline_ids)}")

in_baseline  = []
not_in_map   = []
not_baseline = []

for name in sorted(target_bins):
    sids = name_to_sids.get(name, [])
    if not sids:
        not_in_map.append(name)
    else:
        for sid in sids:
            if sid in baseline_ids:
                in_baseline.append(name)
                break
        else:
            not_baseline.append(name)

print(f"\nBins NOT found in covdat_map (naming mismatch / parse garbage): {len(not_in_map)}")
for n in not_in_map:
    print(f"  {n!r}")

print(f"\nBins found in covdat_map but ALREADY IN BASELINE: {len(in_baseline)}")
for n in in_baseline:
    print(f"  {n}")

print(f"\nBins found in covdat_map and NOT in baseline (genuinely reachable): {len(not_baseline)}")
for n in not_baseline:
    print(f"  {n}")

# Arata distributia pe grupe (GROUP 1 = mcountinhibit, etc.)
print("\n--- mcountinhibit bins specifically ---")
for cat, lst in [("NOT in map", not_in_map), ("already in baseline", in_baseline), ("genuinely reachable", not_baseline)]:
    mc = [n for n in lst if "mcountinhibit" in n.lower()]
    if mc:
        print(f"  {cat}: {mc}")
