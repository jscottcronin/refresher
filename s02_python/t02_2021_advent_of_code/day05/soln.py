import re

import numpy as np


def read_data(filename):
    with open(filename, "r") as f:
        pattern = r"(\d+,\d+) -> (\d+,\d+)"
        matches = re.findall(pattern, f.read())
    data = [[tuple(map(int, i.split(","))) for i in m] for m in matches]
    return data


def init_array(data):
    x_max, y_max = 0, 0
    for p1, p2 in data:
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        if max(x1, x2) > x_max:
            x_max = max(x1, x2)
        if max(y1, y2) > y_max:
            y_max = max(y1, y2)
    return np.zeros((x_max + 1, y_max + 1))


def soln(data, part2=False):
    x = init_array(data)
    for p1, p2 in data:
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        if x1 == x2:
            x[x1, min(y1, y2) : max(y1, y2) + 1] += 1
        if y1 == y2:
            x[min(x1, x2) : max(x1, x2) + 1, y1] += 1
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) == abs(dy) and part2:
            xs = range(x1, x1 + dx + np.sign(dx), np.sign(dx))
            ys = range(y1, y1 + dy + np.sign(dy), np.sign(dy))
            for point in zip(xs, ys):
                x[point] += 1
    return (x > 1).sum().sum()


if __name__ == "__main__":
    fn = "day05/input.txt"
    data = read_data(fn)
    print(soln(data))
    print(soln(data, part2=True))
