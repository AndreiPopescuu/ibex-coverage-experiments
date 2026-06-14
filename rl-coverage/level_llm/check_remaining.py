import pickle, sys, re
from collections import Counter
sys.path.insert(0, '../level9_ops')
sys.path.insert(0, '../level5_real_rtl')
import cov_parser
from starter import BASELINE_PKL, COVDAT, stable_id

hits = set(pickle.load(open(BASELINE_PKL, 'rb')))
s = cov_parser.parse(str(COVDAT))

baseline_ids = {stable_id(h) for h in hits}
remaining = [k for k in s.points if 'v_toggle/' in k and stable_id(k) not in baseline_ids]

def extract_signal(key):
    m = re.search(r'v_toggle/[^o]*o([^\[h]+)', key)
    return m.group(1).strip() if m else key

def classify(k):
    kl = k.lower()
    sig = extract_signal(k)
    # Hardware disabled by config
    if any(p in kl for p in ['rvfi', 'icache', 'pmp', 'debug_mode', 'tmatch',
                               'shadow', 'data_ind', 'mhpm', 'cpuctrl',
                               'debug_ebreak', 'debug_single', 'debug_cause',
                               'debug_csr', 'dret', 'mstatus_err', 'mtvec_err',
                               'pc_mismatch', 'alert', 'crash_dump']):
        return 'hw_disabled'
    # RISC-V architectural: PC/address bits 0,1 always 0
    if re.search(r'(instr_addr|pc_if|pc_id|pc_wb|exception_pc|mepc|boot_addr)\[0\]', sig):
        return 'arch_pc_bit0'
    if re.search(r'(instr_addr|pc_if|pc_id|pc_wb|exception_pc|mepc|boot_addr)\[1\]', sig):
        return 'arch_pc_bit1'
    if re.search(r'(data_addr|data_wdata)_o\[0\]', sig):
        return 'arch_addr_bit0'
    # ALU intermediate values (imd_val) — multi-cycle MUL/DIV internals
    if 'imd_val' in kl:
        return 'alu_intermediate'
    # Counter high bits (mcycle, minstret upper 22 bits)
    if re.search(r'(mcycle|minstret|count)\[(1[0-9]|2[0-9]|3[01])\]', sig):
        return 'counter_high_bits'
    # Writeback data high bits that never toggle
    if re.search(r'result_o\[3[0-1]\]', sig):
        return 'alu_high_bits'
    return 'unknown'

cats = Counter()
unknown_signals = Counter()
for k in remaining:
    c = classify(k)
    cats[c] += 1
    if c == 'unknown':
        sig = extract_signal(k)
        base = re.sub(r'\[\d+\]', '', sig).strip()
        unknown_signals[base] += 1

print(f'Total remaining (uncovered): {len(remaining)}')
print()
print('Breakdown by category:')
for cat, n in cats.most_common():
    print(f'  {cat:<25} {n:>5}  ({100*n/len(remaining):.1f}%)')

unknown_n = cats.get('unknown', 0)
if unknown_n:
    print(f'\nTop signals in "unknown" ({unknown_n} bins):')
    for sig, n in unknown_signals.most_common(15):
        print(f'  {n:>4}x  {sig}')
