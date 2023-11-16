from functools import lru_cache

import numpy as np


def read_data(filename):
    with open(filename, "r") as f:
        data = np.array([int(x) for x in f.read().strip().split(",")])
    return data


def part1(data):
    data = data.copy()
    for _ in range(80):
        data -= 1
        new_births = data == -1
        data[new_births] = 6
        data = np.append(data, [8] * sum(new_births))
    return len(data)


@lru_cache(maxsize=None)
def spawn(day, n):
    if day <= n:
        return 1
    cycles = (day - n - 1) // 7 + 1
    return 1 + sum([spawn(day - n - 1 - 7 * i, 8) for i in range(cycles)])


def part2(data):
    return sum([spawn(256, x) for x in data])


if __name__ == "__main__":
    fn = "day06/input.txt"
    data = read_data(fn)
    print(part1(data))
    print(part2(data))
