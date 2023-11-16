from collections import deque

import numpy as np


def read_data(filename):
    with open(filename, "r") as f:
        return np.array([list(l) for l in f.read().split("\n")]).astype(int)


def soln(data, steps=100):
    x = np.zeros((data.shape[0] + 2, data.shape[1] + 2), dtype=int) - 1
    x[1:-1, 1:-1] = data
    mask = x > -1

    count, step, part1 = 0, 0, 0
    all_flash = x[mask].sum()
    while all_flash != 0:
        flashes = set(zip(*np.where(x == 9)))
        queue = deque(flashes)
        while queue:
            count += 1
            i, j = queue.popleft()
            x[i - 1 : i + 2, j - 1 : j + 2] += 1
            new_flashes = set(zip(*np.where(x == 9)))
            for new_flash in new_flashes:
                if new_flash not in flashes:
                    flashes.add(new_flash)
                    queue.append(new_flash)
            x[~mask] = -1
        x[mask] += 1
        for flash in flashes:
            x[flash] = 0
        all_flash = x[mask].sum()
        step += 1
        if step == steps:
            part1 = count
    return part1, step


if __name__ == "__main__":
    fn = "day11/input.txt"
    data = read_data(fn)
    p1, p2 = soln(data)
    print(p1)
    print(p2)
