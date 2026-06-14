import pickle, sys, re
from collections import defaultdict
sys.path.insert(0, '../level9_ops')
sys.path.insert(0, '../level5_real_rtl')
import cov_parser
from starter import BASELINE_PKL, COVDAT, stable_id

hits = set(pickle.load(open(BASELINE_PKL, 'rb')))
s = cov_parser.parse(str(COVDAT))
baseline_ids = {stable_id(h) for h in hits}
remaining = [k for k in s.points if 'v_toggle/' in k and stable_id(k) not in baseline_ids]

HW_DISABLED = [
    'rvfi', 'icache', 'pmp', 'debug_mode', 'tmatch', 'shadow', 'data_ind',
    'mhpm', 'cpuctrl', 'debug_ebreak', 'debug_single', 'debug_cause',
    'debug_csr', 'dret', 'mstatus_err', 'mtvec_err', 'pc_mismatch',
    'alert', 'crash_dump', 'ic_data_', 'ic_tag_', 'scramble_key',
    'scramble_nonce', 'dscratch', 'dscratc', 'counter_val_upd', 'ic_hit',
    'ic_valid', 'ic_inval', 'ecc_err', 'bus_intg', 'intg_err',
    'bfp_mask', 'bfp_result', 'pack_result', 'sext_result', 'minmax_result',
    'bfp_rev', 'bfp_mask_rev', 'nt_branc', 'predict_branc', 'cnt_branc',
    'nt_branch', 'predict_branch', 'cnt_branch', 'boot_addr',
    'singlebit_result', 'rev_result', 'xperm_result', 'butterfly_result',
    'invbutterfly_result', 'clmul_result', 'multicycle_result',
    'irq_nm', 'irq_pending', 'dummy_instr', 'double_fault',
    'depc_q', 'csr_depc', 'depc_d', 'mcountin',
    'tselect_rdata', 'tselect_q', 'tinfo_rdata', 'tcontrol',
    'tmatch_control', 'tmatch_value',
]

def is_hw_disabled(k):
    kl = k.lower()
    return any(p in kl for p in HW_DISABLED)

def parse_signal_bit(k):
    m = re.search(r'v_toggle/\S+?o(\w[\w_]*)\[(\d+)\]', k)
    if m:
        return m.group(1), int(m.group(2))
    m2 = re.search(r'o([\w_]+(?:\[\d+\])?)h', k)
    if m2:
        return m2.group(1), None
    return None, None

unknown = [k for k in remaining if not is_hw_disabled(k)]

by_signal = defaultdict(list)
for k in unknown:
    sig, bit = parse_signal_bit(k)
    if sig and len(sig) > 2:
        by_signal[sig].append(bit)

print(f"Unknown bins: {len(unknown)}")
print(f"Unique signals: {len(by_signal)}")
print()
print("Signal -> uncovered bits:")
for sig, bits in sorted(by_signal.items(), key=lambda x: -len(x[1])):
    bits_clean = sorted([b for b in bits if b is not None])
    print(f"  {sig:40s} bits: {bits_clean[:16]}{'...' if len(bits_clean)>16 else ''}")
