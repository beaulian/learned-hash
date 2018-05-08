
class Queue(object):
    def __init__(self):
        self.__q = []

    def put(self, k):
        self.__q.append(k)

    def pop(self):
        tmp = self.__q[0]
        del self.__q[0]
        return tmp

    def empty(self):
        return not len(self.__q)

