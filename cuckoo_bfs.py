# coding=utf-8

import math
import queue
import random

DEFAULT_BUCKET_SIZE = 4
DEFAULT_STASH_SIZE = 101
DEFAULT_TIMES = 2.01
DEFAULT_MAX_ITERS = 500
PRIME_DIVISOR = 4294967291
ENTRY_EMPTY = -1
KEY_NOT_FOUND = -1
MAX_BFS_PATH_LEN = 8


make_entry = lambda k, v: (k << 32) + v
get_key = lambda e: e >> 32
get_value = lambda e: e & 0xffffffff
gen_random = lambda: random.randint(1, PRIME_DIVISOR - 1)
alt_mod = lambda x, y: (x * y) >> 32


class SearchSlot(object):
    def __init__(self, index, depth):
        self.index = index
        self.depth = depth
        # check depth
        if not (-1 <= self.depth <= MAX_BFS_PATH_LEN - 1):
            raise ValueError("depth must >= -1 and <= MAX_BFS_PATH_LEN - 1")


class Bucket(object):
    def __init__(self):
        self.__capacity = 0
        self.__b = [ENTRY_EMPTY for _ in range(DEFAULT_BUCKET_SIZE)]

    def occupied(self, slot):
        return self.__b[slot] != ENTRY_EMPTY

    def insert(self, key, value):
        if self.__capacity >= DEFAULT_BUCKET_SIZE:
            return False
        self.__b[self.__capacity] = make_entry(key, value)
        self.__capacity += 1
        return True

    def insert_by_slot(self, slot, key, value):
        if self.__capacity >= DEFAULT_BUCKET_SIZE:
            return
        self.__b[slot] = make_entry(key, value)
        self.__capacity += 1

    def get_by_slot(self, slot):
        return get_key(self.__b[slot]), get_value(self.__b[slot])

    def get(self, key):
        for e in self.__b:
            if get_key(e) == key:
                return get_value(e)
        return KEY_NOT_FOUND

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
        self.__search_depth = 0
        self.__constants_1 = [gen_random(), gen_random()]
        self.__constants_2 = [gen_random(), gen_random()]
        self.__stash_constants = [gen_random(), gen_random()]
        self.__max_iters = max_iters
        self.__stash_size = stash_size
        self.__buckets = [Bucket() for _ in range(self.__num_buckets)]
        self.__stash = [Bucket() for _ in range(self.__stash_size)]

    def insert(self, keys_, values_):
        print('numb_buckets: ', self.__num_buckets, 'stash_size: ', self.__stash_size)
        for i in range(len(keys_)):
            self.__insert(keys_[i], values_[i])
        # for bu in self.__buckets:
        #     print(str(bu))
        return self.__insert_count + self.__stash_count

    def __insert(self, key, value):
        i1 = self.__hash(key, self.__constants_1)
        i2 = self.__hash(key, self.__constants_2)
        # insert
        q = queue.Queue()
        q.put(SearchSlot(i1, 0))
        q.put(SearchSlot(i2, 0))
        while not q.empty():
            x = q.get()
            b = self.__buckets[x.index]
            for slot in range(DEFAULT_BUCKET_SIZE):
                if not b.occupied(slot):
                    self.__search_depth += 1
                    self.__insert_count += 1
                    b.insert_by_slot(slot, key, value)
                    return
                if x.depth < MAX_BFS_PATH_LEN - 1:
                    key, value = b.get_by_slot(slot)
                    q.put(SearchSlot(self.__next_index(key, x.index), x.depth + 1))
        # stash
        # print('enter stash: ', key)
        slot = self.__stash_hash(key, self.__stash_constants)
        if self.__stash[slot].insert(key, value):
            self.__stash_count += 1
            return
        # failed
        print("failed to insert: ", key)

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
        slot = self.__stash_hash(key, self.__stash_constants)
        v3 = self.__stash[slot].get(key)
        if v3 != KEY_NOT_FOUND:
            return v3
        return KEY_NOT_FOUND

    def __hash(self, k, constants_):
        return (constants_[0] ^ k + constants_[1]) % self.__num_buckets

    def __stash_hash(self, k, stash_constants_):
        return (stash_constants_[0] ^ k + stash_constants_[1]) % self.__stash_size

    def load_factor(self):
        return (self.__insert_count + self.__stash_count) / \
               ((self.__num_buckets + self.__stash_size) * 4)

    def insert_count(self):
        return self.__insert_count

    def stash_count(self):
        return self.__stash_count

    def average_search_depth(self):
        return self.__search_depth / self.__num_buckets


def read_keys(file):
    with open(file, "r") as f:
        for line in f:
            yield int(line)


if __name__ == "__main__":
    import time

    N = 19000
    keys = list(read_keys("lognormal.sorted.%d.txt" % N))
    values = [i for i in range(N)]
    cuckoo = Cuckoo(capacity=N, times=0.3)

    start = time.time()
    count = cuckoo.insert(keys, values)
    end = time.time()
    print("average insert time: %.2f us" % (((end - start) / N) * 1000000))

    print("load factor: ", cuckoo.load_factor())
    print("insert count: ", cuckoo.insert_count())
    print("stash count: ", cuckoo.stash_count())
    print("sum count: ", count)

    start = time.time()
    results = cuckoo.find(keys)
    end = time.time()
    print("average find time: %.2f us" % (((end - start) / N) * 1000000))

    hits = 0
    for i in range(len(results)):
        if results[i] == values[i]:
            hits += 1
    print("hits rate: ", hits / count)
