import os
from collections import defaultdict

raw = 'raw_data/ml-100k/u.data'
rows = []
with open(raw) as f:
    for line in f:
        u, i, r, t = line.rstrip('\n').split('\t')
        rows.append((u, i, int(r), int(t)))

def filter_k_core(rows, user_core=5, item_core=5):
    while True:
        uc = defaultdict(int); ic = defaultdict(int)
        for u,i,r,t in rows:
            uc[u]+=1; ic[i]+=1
        du = {u for u,c in uc.items() if c<user_core}
        di = {i for i,c in ic.items() if c<item_core}
        if not du and not di:
            break
        rows = [x for x in rows if x[0] not in du and x[1] not in di]
    return rows

rows = filter_k_core(rows, 5, 5)
rows.sort(key=lambda x: x[3])  # by timestamp
users = len({x[0] for x in rows}); items = len({x[1] for x in rows})
print(f'after 5-core: {len(rows)} interactions, {users} users, {items} items')

seq = defaultdict(list)
for u,i,r,t in rows:
    seq[u].append(i)

os.makedirs('data/ML100K', exist_ok=True)
with open('data/ML100K/user_sequence.txt','w') as out:
    for u, its in seq.items():
        out.write(u + ' ' + ' '.join(its) + '\n')
print('wrote data/ML100K/user_sequence.txt')
