import sys
sys.path.insert(0, '../level9_ops')
sys.path.insert(0, '../level5_real_rtl')
import starter

baseline_ids, covdat_map, target_bins = starter.setup()
nops = [(10, 0, 0, 0, 2)] * 500
r = starter.run_and_check(nops, baseline_ids, covdat_map, target_bins)
print('new_total:', r['new_total'])
print('new_target:', r['new_target'])
if r['new_total'] > 0:
    print('signals:', r['signals'][:10])
