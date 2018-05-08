# coding=utf-8

import math
import time
import random

from queue import Queue

DEFAULT_BUCKET_SIZE = 4
DEFAULT_STASH_SIZE = 101
DEFAULT_TIMES = 2.01
DEFAULT_MAX_ITERS = 500
PRIME_DIVISOR = 4294967291
ENTRY_EMPTY = -1
KEY_NOT_FOUND = -1
N = 19000
MAX_BFS_PATH_LEN = int(math.ceil(
    math.log(N/DEFAULT_BUCKET_SIZE/2 - N/DEFAULT_BUCKET_SIZE/(2*DEFAULT_BUCKET_SIZE) + 1)
))
NULL = -2


make_entry = lambda k, v: (k << 32) + v
get_key = lambda e: e >> 32
get_value = lambda e: e & 0xffffffff
gen_random = lambda: random.randint(1, 2 ** 32 - 1)
alt_mod = lambda x, y: (x * y) >> 32


class SearchSlot(object):
    def __init__(self, index, pathcode, depth):
        self.index = index
        self.pathcode = pathcode
        self.depth = depth
        # check depth
        if not (-1 <= self.depth <= MAX_BFS_PATH_LEN - 1):
            raise ValueError("depth must >= -1 and <= MAX_BFS_PATH_LEN - 1")


class CuckooPath(object):
    def __init__(self, index, slot, key, value):
        self.index = index
        self.slot = slot
        self.key = key
        self.value = value


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
            return False
        self.__b[slot] = make_entry(key, value)
        self.__capacity += 1
        return True

    def get_by_slot(self, slot):
        return get_key(self.__b[slot]), get_value(self.__b[slot])

    def erase_by_slot(self, slot):
        self.__b[slot] = ENTRY_EMPTY
        self.__capacity -= 1

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
    def __init__(self, capacity, times=DEFAULT_TIMES):
        self.__num_buckets = int(math.ceil(capacity * times))
        self.__insert_count = 0
        self.__search_depth = 0
        self.__buckets = [Bucket() for _ in range(self.__num_buckets)]
        self.__slot_search_time = 0.0

    def insert(self, keys_, values_):
        print('numb_buckets: ', self.__num_buckets)
        for i in range(len(keys_)):
            self.__insert(keys_[i], values_[i])
        # for bu in self.__buckets:
        #     print(str(bu))
        return self.__insert_count

    def __insert(self, key, value):
        print(key)
        i1, i2 = self.__gen_i1_i2(key)
        # search path
        depth, cuckoo_path = self.__find_cuckoo_path(i1, i2)
        if depth >= 0:
            # for i in range(depth+1):
            #     print("cuckoo path", depth, cuckoo_path[i].index, cuckoo_path[i].slot, cuckoo_path[i].key)
            self.__cuckoo_path_move(cuckoo_path, depth)
            assert self.__buckets[cuckoo_path[0].index].insert_by_slot(cuckoo_path[0].slot, key, value)
            self.__insert_count += 1
            return

    def __find_cuckoo_path(self, i1, i2):
        cuckoo_path = [CuckooPath(NULL, NULL, NULL, NULL) for _ in range(MAX_BFS_PATH_LEN)]
        start = time.time()
        x = self.__slot_search(i1, i2)
        end = time.time()
        self.__slot_search_time += (end - start)
        if x.depth == -1:
            return -1, []
        for i in range(x.depth, -1, -1):
            cuckoo_path[i].slot = x.pathcode % DEFAULT_BUCKET_SIZE
            x.pathcode = int(x.pathcode / DEFAULT_BUCKET_SIZE)
        # fill
        first = cuckoo_path[0]
        first.index = i1 if x.pathcode == 0 else i2
        b = self.__buckets[first.index]
        first.key, first.value = b.get_by_slot(first.slot)
        # fill others
        for i in range(1, x.depth + 1):
            curr = cuckoo_path[i]
            prev = cuckoo_path[i - 1]
            curr.index = self.__next_index(prev.key, prev.index)
            curr.key, curr.value = self.__buckets[curr.index].get_by_slot(curr.slot)
        return x.depth, cuckoo_path

    def __slot_search(self, i1, i2):
        # insert
        q = Queue()
        q.put(SearchSlot(i1, 0, 0))
        q.put(SearchSlot(i2, 1, 0))
        while not q.empty():
            x = q.pop()
            b = self.__buckets[x.index]
            starting_slot = x.pathcode % DEFAULT_BUCKET_SIZE
            for i in range(DEFAULT_BUCKET_SIZE):
                slot = (starting_slot + i) % DEFAULT_BUCKET_SIZE
                if not b.occupied(slot):
                    x.pathcode = x.pathcode * DEFAULT_BUCKET_SIZE + slot
                    # print("1: ", x.index, slot, x.depth, x.pathcode)
                    return x
                print("2: ", x.index, slot, x.depth, x.pathcode)
                if x.depth < MAX_BFS_PATH_LEN - 1:
                    key, value = b.get_by_slot(slot)
                    q.put(SearchSlot(self.__next_index(key, x.index),
                                     x.pathcode * DEFAULT_BUCKET_SIZE + slot, x.depth + 1))
        return SearchSlot(0, 0, -1)

    def __cuckoo_path_move(self, cuckoo_path, depth):
        # print(cuckoo_path[0].index, cuckoo_path[0].slot, cuckoo_path[0].key)
        self.__search_depth += depth
        if depth > 0:
            from_ = NULL
            for depth_ in range(depth, 0, -1):
                from_ = cuckoo_path[depth_ - 1]
                to_ = cuckoo_path[depth_]
                # print("depth_: ", depth_)
                assert self.__buckets[to_.index].insert_by_slot(to_.slot, from_.key, from_.value)
                self.__buckets[from_.index].erase_by_slot(from_.slot)

    def __next_index(self, key, previous_index):
        i1, i2 = self.__gen_i1_i2(key)
        return i2 if previous_index == i1 else i1

    def find(self, keys_, iter_=False):
        results_ = map(self.__find, keys_)
        return results if iter_ else list(results_)

    def __find(self, key):
        i1, i2 = self.__gen_i1_i2(key)
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
        #     return v3
        return KEY_NOT_FOUND

    def __gen_i1_i2(self, key):
        hash_ = self.__hash(key)
        i1 = hash_ % self.__num_buckets
        i2 = self.__hash(key ^ hash_) % self.__num_buckets
        return i1, i2

    @staticmethod
    def __hash(value):
        value ^= value >> 16
        value *= 0x85ebca6b
        value ^= value >> 13
        value *= 0xc2b2ae35
        value ^= value >> 16
        return value

    def load_factor(self):
        return self.__insert_count / (self.__num_buckets * 4)

    def insert_count(self):
        return self.__insert_count

    def average_search_depth(self):
        return self.__search_depth / self.__num_buckets

    def slot_search_time(self):
        return self.__slot_search_time * 1000000


def read_keys(file):
    with open(file, "r") as f:
        for line in f:
            yield int(line)


def generate_random_keys(n_):
    for _ in range(n_):
        yield random.randint(1, 2**32 - 1)


if __name__ == "__main__":
    # keys = list(read_keys("lognormal.sorted.%d.txt" % N))
    keys = list(generate_random_keys(N))
    values = [i for i in range(N)]
    cuckoo = Cuckoo(capacity=N/DEFAULT_BUCKET_SIZE, times=0.99)

    start = time.time()
    count = cuckoo.insert(keys, values)
    end = time.time()
    print("average insert time: %.2f us" % (((end - start) / N) * 1000000))
    print("average slot search time: %.2f us" % (cuckoo.slot_search_time() / N))

    print("load factor: ", cuckoo.load_factor())
    print("insert count: ", cuckoo.insert_count())
    print("average search depth: %.2f" % cuckoo.average_search_depth())

    start = time.time()
    results = cuckoo.find(keys)
    end = time.time()
    print("average find time: %.2f us" % (((end - start) / N) * 1000000))

    hits = 0
    for i in range(len(results)):
        if results[i] == values[i]:
            hits += 1
    print("hits rate: ", hits / count)
