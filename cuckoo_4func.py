# coding=utf-8

import math
import random
from collections import Counter

DEFAULT_STASH_SIZE = 101
DEFAULT_TIMES = 2.01
DEFAULT_MAX_ITERS = 190000
PRIME_DIVISOR = 4294967291
ENTRY_EMPTY = -1
KEY_NOT_FOUND = -1


make_entry = lambda k, v: (k << 32) + v
get_key = lambda e: e >> 32
get_value = lambda e: e & 0xffffffff
gen_random = lambda: random.randint(1, PRIME_DIVISOR - 1)
alt_mod = lambda x, y: (x * y) >> 32


class Cuckoo(object):
    def __init__(self, capacity, times=DEFAULT_TIMES, max_iters=DEFAULT_MAX_ITERS):
        self.__num_buckets = int(math.ceil(capacity * times))
        self.__insert_count = 0
        self.__conflict_count = 0
        self.__constants_1 = [gen_random(), gen_random()]
        self.__constants_2 = [gen_random(), gen_random()]
        self.__constants_3 = [gen_random(), gen_random()]
        self.__constants_4 = [gen_random(), gen_random()]
        self.__max_iters = max_iters
        self.__buckets = [ENTRY_EMPTY for _ in range(self.__num_buckets)]
        self.search_count = []
        # self.__rfile = open('dataset.txt', 'w')

    # def __del__(self):
    #     self.__rfile.close()

    def insert(self, keys_, values_):
        print('numb_buckets: ', self.__num_buckets)
        for i in range(len(keys_)):
            self.__insert(keys_[i], values_[i])
        # for bu in self.__buckets:
        #     print(str(bu))
        return self.__insert_count

    def __insert(self, key, value):
        count = 1

        indices = []
        for i in range(1, 5):
            i = self.__hash(key, getattr(self, f"_Cuckoo__constants_{i}"))
            b = self.__buckets[i]
            if b == ENTRY_EMPTY:
                self.__buckets[i] = make_entry(key, value)
                self.__insert_count += 1
                self.search_count.append(count)
                return True
            indices.append(i)

        i = random.choice(indices)
        b = self.__buckets[i]
        for _ in range(self.__max_iters):
            count += 1
            self.__conflict_count += 1
            entry = make_entry(key, value)
            self.__buckets[i] = entry
            key, value = get_key(b), get_value(b)
            i = self.__next_index(key, i)
            b = self.__buckets[i]
            if b == ENTRY_EMPTY:
                self.__buckets[i] = make_entry(key, value)
                # self.__rfile.write("%d %d\n" % (key, i))
                self.search_count.append(count)
                self.__insert_count += 1
                return True

        return False

    def __next_index(self, key, previous_index):
        indices = []
        for i in range(1, 5):
            i = self.__hash(key, getattr(self, f"_Cuckoo__constants_{i}"))
            indices.append(i)
        for i in range(4):
            if indices[i] == previous_index:
                next_index = indices[i+1] if i < 3 else indices[0]
                return next_index

    def find(self, keys_, iter_=False):
        results_ = map(self.__find, keys_)
        return results if iter_ else list(results_)

    def __find(self, key):
        for i in range(1, 5):
            i = self.__hash(key, getattr(self, f"_Cuckoo__constants_{i}"))
            v = self.__buckets[i]
            if key == get_key(v):
                return get_value(v)

        return KEY_NOT_FOUND

    def __hash(self, k, constants_):
        return (constants_[0] ^ k + constants_[1]) % self.__num_buckets

    def load_factor(self):
        return self.__insert_count / self.__num_buckets

    def insert_count(self):
        return self.__insert_count

    def conflict_count(self):
        return self.__conflict_count


def read_keys(file):
    with open(file, "r") as f:
        for line in f:
            yield int(line)


if __name__ == "__main__":
    import time

    N = 19000
    keys = list(read_keys("lognormal.sorted.%d.txt" % N))
    values = [i for i in range(N)]
    cuckoo = Cuckoo(capacity=N, times=1.05)
    start = time.time()
    count = cuckoo.insert(keys, values)
    end = time.time()
    print("average insert time: %.2f us" % (((end - start) / N) * 1000000))

    print("load factor: ", cuckoo.load_factor())
    print("insert count: ", cuckoo.insert_count())
    print("sum count: ", count)
    # print("conflict count: ", cuckoo.conflict_count())
    print("search count: ", Counter(cuckoo.search_count))

    start = time.time()
    results = cuckoo.find(keys)
    end = time.time()
    print("average find time: %.2f us" % (((end - start) / N) * 1000000))

    hits = 0
    for i in range(len(results)):
        if results[i] == values[i]:
            hits += 1

    print("hits rate: ", hits / count)
