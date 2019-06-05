# coding=utf-8

import math
import random
from collections import Counter

DEFAULT_BUCKET_SIZE = 4
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


class Bucket(object):
    def __init__(self):
        self.__capacity = 0
        self.__b = [ENTRY_EMPTY for _ in range(DEFAULT_BUCKET_SIZE)]

    def insert(self, key, value):
        if self.__capacity >= len(self.__b):
            return False
        self.__b[self.__capacity] = make_entry(key, value)
        self.__capacity += 1
        return True

    def get(self, key):
        for e in self.__b:
            if get_key(e) == key:
                return get_value(e)
        return KEY_NOT_FOUND

    def swap(self, key, value):
        i = random.randint(0, 3)
        entry = make_entry(key, value)
        entry, self.__b[i] = self.__b[i], entry
        return get_key(entry), get_value(entry)

    def __len__(self):
        return self.__capacity

    def __contains__(self, key):
        return self.get(key) != -1

    def __repr__(self):
        return "<Bucket: b=" + str(self.__b) + ">"

    def __str__(self):
        return self.__repr__()


class Cuckoo(object):
    def __init__(self, capacity, times=DEFAULT_TIMES,
                 max_iters=DEFAULT_MAX_ITERS, stash_size=DEFAULT_STASH_SIZE):
        self.__num_buckets = int(math.ceil(capacity * times))
        self.__stash_count = 0
        self.__insert_count = 0
        self.__conflict_count = 0
        self.__constants_1 = [gen_random(), gen_random()]
        self.__constants_2 = [gen_random(), gen_random()]
        self.__stash_constants = [gen_random(), gen_random()]
        self.__max_iters = max_iters
        self.__stash_size = stash_size
        self.__buckets = [Bucket() for _ in range(self.__num_buckets)]
        # self.__stash = [Bucket() for _ in range(self.__stash_size)]
        self.search_count = []
        # self.__rfile = open('dataset.txt', 'w')

    # def __del__(self):
    #     self.__rfile.close()

    def insert(self, keys_, values_):
        print('numb_buckets: ', self.__num_buckets, 'stash_size: ', self.__stash_size)
        for i in range(len(keys_)):
            self.__insert(keys_[i], values_[i])
        # for bu in self.__buckets:
        #     print(str(bu))
        return self.__insert_count + self.__stash_count

    def __insert(self, key, value):
        count = 1
        i1 = self.__hash(key, self.__constants_1)
        b1 = self.__buckets[i1]
        # try insert
        if b1.insert(key, value):
            # self.__rfile.write("%d %d\n" % (key, i1))
            self.__insert_count += 1
            self.search_count.append(count)
            return True
        i2 = self.__hash(key, self.__constants_2)
        b2 = self.__buckets[i2]
        if b2.insert(key, value):
            # self.__rfile.write("%d %d\n" % (key, i2))
            self.__insert_count += 1
            self.search_count.append(count)
            return True
        # iter
        # print('enter iter: ', key)

        i = random.choice([i1, i2])
        b = self.__buckets[i]
        for _ in range(self.__max_iters):
            count += 1
            self.__conflict_count += 1
            key, value = b.swap(key, value)
            i = self.__next_index(key, i)
            b = self.__buckets[i]
            if b.insert(key, value):
                # self.__rfile.write("%d %d\n" % (key, i))
                self.search_count.append(count)
                self.__insert_count += 1
                return True
        # stash
        # print('enter stash: ', key)
        # slot = self.__stash_hash(key, self.__stash_constants)
        # if self.__stash[slot].insert(key, value):
            # self.__rfile.write("%d %d\n" % (key, slot + self.__num_buckets))
        #    self.__stash_count += 1
        #    return True

        return False

    def __next_index(self, key, previous_index):
        i1, i2 = self.__hash(key, self.__constants_1), self.__hash(key, self.__constants_2)
        return i2 if previous_index == i1 else i1

    def find(self, keys_, iter_=False):
        results_ = map(self.__find, keys_)
        return results if iter_ else list(results_)

    def __find(self, key):
        i1, i2 = self.__hash(key, self.__constants_1), self.__hash(key, self.__constants_2)
        # i1
        v1 = self.__buckets[i1].get(key)
        if v1 != KEY_NOT_FOUND:
            return v1
        # i2
        v2 = self.__buckets[i2].get(key)
        if v2 != KEY_NOT_FOUND:
            return v2
        # stash
        # slot = self.__stash_hash(key, self.__stash_constants)
        # v3 = self.__stash[slot].get(key)
        # if v3 != KEY_NOT_FOUND:
        #    return v3
        return KEY_NOT_FOUND

    def __hash(self, k, constants_):
        return (constants_[0] ^ k + constants_[1]) % self.__num_buckets

    # def __stash_hash(self, k, stash_constants_):
    #     return (stash_constants_[0] ^ k + stash_constants_[1]) % self.__stash_size

    def load_factor(self):
        return (self.__insert_count + self.__stash_count) / \
               ((self.__num_buckets + self.__stash_size) * 4)

    def insert_count(self):
        return self.__insert_count

    def stash_count(self):
        return self.__stash_count

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
    cuckoo = Cuckoo(capacity=N/DEFAULT_BUCKET_SIZE, times=1.05)
    start = time.time()
    count = cuckoo.insert(keys, values)
    end = time.time()
    print("average insert time: %.2f us" % (((end - start) / N) * 1000000))

    print("load factor: ", cuckoo.load_factor())
    print("insert count: ", cuckoo.insert_count())
    print("stash count: ", cuckoo.stash_count())
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
