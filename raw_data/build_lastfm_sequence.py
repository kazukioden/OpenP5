import os
from collections import defaultdict

# hetrec2011-lastfm-2k: user tags artists with timestamps. Use (user,artist)
# first-tag time to order each user's distinct artists into a sequence.
src = 'raw_data/lastfm/user_taggedartists-timestamps.dat'
first = {}  # (u,a) -> min timestamp
with open(src, encoding='latin-1') as f:
    next(f)  # header
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < 4: continue
        u, a, _, t = p[0], p[1], p[2], int(p[3])
        k = (u, a)
        if k not in first or t < first[k]:
            first[k] = t

rows = [(u, a, t) for (u, a), t in first.items()]

def kcore(rows, uc=5, ic=5):
    while True:
        cu, ci = defaultdict(int), defaultdict(int)
        for u,a,t in rows: cu[u]+=1; ci[a]+=1
        du = {u for u,c in cu.items() if c<uc}
        di = {a for a,c in ci.items() if c<ic}
        if not du and not di: break
        rows = [x for x in rows if x[0] not in du and x[1] not in di]
    return rows

rows = kcore(rows, 5, 5)
rows.sort(key=lambda x: x[2])
users = len({x[0] for x in rows}); items = len({x[1] for x in rows})
print(f'after 5-core: {len(rows)} interactions, {users} users, {items} items, sparsity={1-len(rows)/(users*items):.4f}')

seq = defaultdict(list)
for u,a,t in rows: seq[u].append(a)
os.makedirs('data/LastFM', exist_ok=True)
with open('data/LastFM/user_sequence.txt','w') as out:
    for u,arts in seq.items():
        out.write(u + ' ' + ' '.join(arts) + '\n')
print('wrote data/LastFM/user_sequence.txt')
