from collections import defaultdict
from itertools import product
import numpy as np
import math
import heapq


def read_data(filename):
    with open(filename, "r") as f:
        data = [[int(i) for i in list(l)] for l in f.read().split("\n")]
    return np.array(data)


def part1(arr):
    start = (0, 0)
    end = (arr.shape[0] - 1, arr.shape[1] - 1)

    dist = defaultdict(lambda: math.inf, {start: 0})
    pq = [(0, start)]
    visited = set()

    while pq:
        x, p = heapq.heappop(pq)
        if p == end:
            return dist[end]
        if p in visited:
            continue

        visited.add(p)
        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= p[0] + i < arr.shape[0] and 0 <= p[1] + j < arr.shape[1]:
                neighbor = (p[0] + i, p[1] + j)
                new_x = x + arr[neighbor]
                if new_x < dist[neighbor]:
                    dist[neighbor] = new_x
                    heapq.heappush(pq, (new_x, neighbor))
    return math.inf


def part2(arr):
    arr = np.concatenate([arr + i for i in range(5)], axis=1)
    arr = np.concatenate([arr + i for i in range(5)], axis=0)
    while (arr > 9).any():
        arr[arr > 9] -= 9
    return part1(arr)


if __name__ == "__main__":
    fn = "day15/input.txt"
    data = read_data(fn)
    print(part1(data))
    print(part2(data))
