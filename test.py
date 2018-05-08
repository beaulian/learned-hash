
from collections import defaultdict


# def Murmur3__hash(value):
#     value ^= value >> 16
#     value *= 0x85ebca6b
#     value ^= value >> 13
#     value *= 0xc2b2ae35
#     value ^= value >> 16
#     return value

#
# with open('lognormal.sorted.19000.txt', 'r') as f:
#     with open('result.txt', 'w') as f1:
#         for line in f:
#             f1.write("%d\n" % Murmur3__hash(int(line.strip())))


d1 = defaultdict(int)
d2 = defaultdict(int)

with open('result.txt', 'r') as f:
    for line in f:
        k1, k2 = line.strip().split()
        d1[k1] += 1
        d2[k2] += 1

print(max(d1.values()), max(d2.values()))
