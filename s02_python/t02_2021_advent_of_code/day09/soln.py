from collections import deque
from functools import reduce

import numpy as np


def read_data(filename):
    with open(filename, "r") as f:
        data = [[int(c) for c in line] for line in f.read().split("\n")]
    return np.array(data)


def get_basins(x):
    neighbors = np.zeros((4, x.shape[0] + 2, x.shape[1] + 2), dtype=int) + 10
    for i, (j, k) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        neighbors[i, 1 - j : x.shape[0] - j + 1, 1 - k : x.shape[1] - k + 1] = x
    basins = (neighbors[:, 1:-1, 1:-1] > x).all(axis=0)
    return basins


def bfs(arr, point):
    visited = set()
    queue = deque()
    queue.append(point)
    visited.add(point)
    while queue:
        x, y = queue.popleft()
        for p in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if p not in visited and arr[p] != 9:
                queue.append(p)
                visited.add(p)
    return visited


def part1(x):
    basins = get_basins(x)
    return (x[basins] + 1).sum()


def part2(x):
    basins = [(i + 1, j + 1) for i, j in np.argwhere(get_basins(x))]
    x2 = np.zeros((x.shape[0] + 2, x.shape[1] + 2), dtype=int) + 9
    x2[1:-1, 1:-1] = x
    top3 = sorted([len(bfs(x2, basin)) for basin in basins])[-3:]
    return reduce(lambda x, y: x * y, top3)


if __name__ == "__main__":
    fn = "day09/input.txt"
    data = read_data(fn)
    print(part1(data))
    print(part2(data))
