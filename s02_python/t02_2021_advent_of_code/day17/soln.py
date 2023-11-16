from itertools import product
import math

import numpy as np


def part1(target):
    return sum(range(-target[2]))


def part2(target):
    x_min, x_max, y_min, y_max = target
    max_steps = (-y_min) * 2
    points = set()

    offsets = np.arange(max_steps).cumsum()
    x_stop_min = np.argmax(offsets >= x_min)
    x_stop_max = np.argmin(offsets <= x_max) - 1

    for step in range(1, max_steps + 1):
        o = offsets[step - 1]
        x_lower = math.ceil((x_min + o) / step) if step < x_stop_max else x_stop_min
        x_upper = (x_max + o) // step if step < x_stop_max else x_stop_max
        y_lower = math.ceil((-y_max - o) / step)
        y_upper = (-y_min - o) // step

        x = list(range(x_lower, x_upper + 1, 1))
        y = [-1 * i for i in list(range(y_lower, y_upper + 1, 1))]
        points.update(list(product(x, y)))
    return len(points)


if __name__ == "__main__":
    # target = (20, 30, -10, -5)    # dev
    target = (117, 164, -140, -89)  # prd
    print(part1(target))
    print(part2(target))
