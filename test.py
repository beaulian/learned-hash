
from collections import defaultdict


d = defaultdict(int)

with open('result.txt', 'r') as f:
    for line in f:
        k1, k2 = line.strip().split()
        d[k1] += 1
        d[k2] += 1

print(max(d.values()))
