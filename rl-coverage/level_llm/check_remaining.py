import pickle, sys
sys.path.insert(0, '../level9_ops')
sys.path.insert(0, '../level5_real_rtl')
import cov_parser
from starter import BASELINE_PKL, COVDAT, stable_id

hits = set(pickle.load(open(BASELINE_PKL, 'rb')))
s = cov_parser.parse(str(COVDAT))

baseline_ids = {stable_id(h) for h in hits}
remaining = [k for k in s.points if 'v_toggle/' in k and stable_id(k) not in baseline_ids]

DISABLED_PATTERNS = [
    'rvfi', 'icache', 'pmp', 'debug_mode', 'tmatch',
    'shadow', 'data_ind', 'mhpm', 'cpuctrl',
    'debug_ebreak', 'debug_single', 'debug_cause', 'debug_csr',
    'dret', 'mstatus_err', 'mtvec_err', 'csr_shadow',
    'pc_mismatch', 'wb_exception',
]

disabled = [k for k in remaining if any(p in k.lower() for p in DISABLED_PATTERNS)]
unknown = [k for k in remaining if k not in disabled]

print(f'Total remaining (uncovered): {len(remaining)}')
print(f'Clearly disabled HW:         {len(disabled)} ({100*len(disabled)/len(remaining):.1f}%)')
print(f'Unknown/possibly reachable:  {len(unknown)} ({100*len(unknown)/len(remaining):.1f}%)')
print()
print('Sample of unknown bins:')
for k in unknown[:20]:
    print(' ', k)
