import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
h, d = rows[0], rows[3]
m = dict(zip(h, d))
pats = ('conflicts_shared', 'wavefronts_shared', 'wavefronts_mem_shared',
        'op_shared_ld', 'op_shared_st', 'inst_executed.sum',
        'op_global_ld', 'op_global_st', 'sectors')
for k in sorted(m):
    if any(p in k for p in pats) and not k.startswith('device__'):
        v = m[k]
        if v:
            print(k, '=', v)
