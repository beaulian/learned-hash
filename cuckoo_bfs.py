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
        self.__rfile = open('result.txt', 'w')

    def insert(self, keys_, values_):
        print('numb_buckets: ', self.__num_buckets, 'stash_size: ', self.__stash_size)
        for i in range(len(keys_)):
            self.__insert(keys_[i], values_[i])
        # for bu in self.__buckets:
        #     print(str(bu))
        return self.__insert_count + self.__stash_count

    def __insert(self, key, value):
        # print(key)
        i1 = self.__hash(key, self.__constants_1)
        i2 = self.__hash(key, self.__constants_2)
        self.__rfile.write("%d %d\n" % (i1, i2))
        # search path
        depth, cuckoo_path = self.__find_cuckoo_path(i1, i2)
        if depth >= 0:
            # for i in range(depth+1):
            #     print("cuckoo path", depth, cuckoo_path[i].index, cuckoo_path[i].slot, cuckoo_path[i].key)
            self.__cuckoo_path_move(cuckoo_path, depth)
            assert self.__buckets[cuckoo_path[0].index].insert_by_slot(cuckoo_path[0].slot, key, value)
            self.__insert_count += 1
            return
        # stash
        # print('enter stash: ', key)
        # slot = self.__stash_hash(key, self.__stash_constants)
        # if self.__stash[slot].insert(key, value):
        #     self.__stash_count += 1
        #     return
        # failed
        # print("failed to insert: ", key)

    def __find_cuckoo_path(self, i1, i2):
        cuckoo_path = [CuckooPath(NULL, NULL, NULL, NULL) for _ in range(MAX_BFS_PATH_LEN)]
        x = self.__slot_search(i1, i2)
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
        q = queue.Queue()
        q.put(SearchSlot(i1, 0, 0))
        q.put(SearchSlot(i2, 1, 0))
        while not q.empty():
            x = q.get()
            b = self.__buckets[x.index]
            starting_slot = x.pathcode % DEFAULT_BUCKET_SIZE
            for i in range(DEFAULT_BUCKET_SIZE):
                slot = (starting_slot + i) % DEFAULT_BUCKET_SIZE
                if not b.occupied(slot):
                    x.pathcode = x.pathcode * DEFAULT_BUCKET_SIZE + slot
                    # print("1: ", x.index, slot, x.depth, x.pathcode)
                    return x
                # print("2: ", x.index, slot, x.depth, x.pathcode)
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
        #     return v3
        return KEY_NOT_FOUND

    def __hash(self, k, constants_):
        return (constants_[0] ^ k + constants_[1]) % self.__num_buckets

    def __stash_hash(self, k, stash_constants_):
        return (stash_constants_[0] ^ k + stash_constants_[1]) % self.__stash_size

    def load_factor(self):
        return self.__insert_count / (self.__num_buckets * 4)

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


def generate_random_keys(n_):
    for _ in range(n_):
        yield random.randint(1, 2**32 - 1)


if __name__ == "__main__":
    import time

    N = 19
    keys = list(read_keys("lognormal.sorted.%d.txt" % N))
    # keys = list(generate_random_keys(N))
    values = [i for i in range(N)]
    cuckoo = Cuckoo(capacity=N/4, times=2)

    start = time.time()
    count = cuckoo.insert(keys, values)
    end = time.time()
    print("average insert time: %.2f us" % (((end - start) / N) * 1000000))

    print("load factor: ", cuckoo.load_factor())
    print("insert count: ", cuckoo.insert_count())
    print("stash count: ", cuckoo.stash_count())
    print("sum count: ", count)
    print("average search depth: %.2f" % cuckoo.average_search_depth())

    start = time.time()
    results = cuckoo.find(keys)
    end = time.time()
    print("average find time: %.2f us" % (((end - start) / N) * 1000000))

    hits = 0
    for i in range(len(results)):
        if results[i] == values[i]:
            hits += 1
        else:
            print("result: ", i)
    print("hits rate: ", hits / count)
