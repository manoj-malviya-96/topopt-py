import time

from matplotlib import pyplot as plt

from core.optimize import TopOptConfig, run_optimization


def benchmark(func, iterator):
    times = []
    nums = []

    while iterator.nCalls < 10:
        start = time.time()
        args, n = iterator.next()
        res = func(args)
        end = time.time()
        nums.append(n)
        times.append(end - start)

    plt.plot(nums, times, 'o')
    plt.plot(nums, times)
    plt.xlabel('nums')
    plt.ylabel('times')
    plt.show()


class TopOptConfigIterator:
    def __init__(self):
        self.config = TopOptConfig()
        self.nCalls = 0

    def next(self):
        print("nCalls: ", self.nCalls)
        if self.nCalls >= 0:
            self.config.nelx = int(1.2 * self.config.nelx)
            self.config.nely = int(1.2 * self.config.nely)
        self.nCalls += 1
        return self.config, self.config.nely * self.config.nelx


top_iter = TopOptConfigIterator()
benchmark(run_optimization, top_iter)
