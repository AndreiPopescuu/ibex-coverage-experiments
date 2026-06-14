import sys, pickle
sys.path.insert(0, '../level5_real_rtl')
sys.path.insert(0, '../level9_ops')
import cov_parser
from starter import stable_id, BASELINE_PKL, COVDAT

hits = pickle.load(open(BASELINE_PKL, 'rb'))
s = cov_parser.parse(str(COVDAT))
tog_keys = [k for k in s.points if 'v_toggle/' in k]
baseline_ids = {stable_id(k) for k in hits}
covdat_ids = {stable_id(k) for k in tog_keys}
print('baseline ids:', len(baseline_ids))
print('covdat ids:', len(covdat_ids))
print('baseline in covdat:', len(baseline_ids & covdat_ids))
print('match pct:', round(100 * len(baseline_ids & covdat_ids) / len(covdat_ids), 2))
only_in_baseline = baseline_ids - covdat_ids
print('in baseline but NOT in covdat:', len(only_in_baseline))
