from dataclasses import dataclass
import json

import numpy as np


class Sand:
    def __init__(self, data):
        self.point = (0, 500)
        self.downleft = None
        self.downright = None
        self.down = None
        self.possible = None
        self.get_surroundings(data)

    def get_surroundings(self, data):
        self.downleft = (self.point[0] + 1, self.point[1] - 1)
        self.downright = (self.point[0] + 1, self.point[1] + 1)
        self.down = (self.point[0] + 1, self.point[1])
        self.possible = "." in [
            data[self.down],
            data[self.downleft],
            data[self.downright],
        ]

    def flow(self, data):
        keep_going = True
        while self.possible:
            if data[self.down] == ".":
                self.point = self.down
            elif data[self.downleft] == ".":
                self.point = self.downleft
            elif data[self.downright] == ".":
                self.point = self.downright
            else:
                raise ValueError("should not be here")
            try:
                self.get_surroundings(data)
            except IndexError:
                keep_going = False
                break
        data[self.point] = "o"
        return data, keep_going


def viz(data):
    print(data[:, 493:504])


def get_range_for_value(x):
    if x >= 0:
        return list(range(0, x + 1, 1))
    else:
        return list(range(0, x - 1, -1))


def get_points_between_vertices(u, v):
    rocks = set([u, v])
    for dx in get_range_for_value(v[0] - u[0]):
        rocks.add((u[0] + dx, u[1]))
    for dy in get_range_for_value(v[1] - u[1]):
        rocks.add((u[0], u[1] + dy))
    return rocks


with open("day_14/data.txt") as f:
    rocks = set()
    for line in f:
        vertices = [tuple(map(int, v.split(","))) for v in line.strip().split(" -> ")]
        for i in range(1, len(vertices)):
            u = vertices[i - 1]
            v = vertices[i]
            rocks = rocks | get_points_between_vertices(u, v)
    max_x = 0
    max_y = 0
    for rock in rocks:
        if rock[0] > max_x:
            max_x = rock[0]
        if rock[1] > max_y:
            max_y = rock[1]

    # Part 1
    data = np.full((max_y + 2, max_x + 100), ".")
    data[0, 500] = "+"
    for rock in rocks:
        data[rock[1], rock[0]] = "#"
    i = 1
    keep_going = True
    while keep_going:
        s = Sand(data)
        data, keep_going = s.flow(data)
        i += 1

    viz(data)
    print(f"Part 1: {i - 2}")

    ## Part 2
    data = np.full((max_y + 3, max_x + 200), ".")
    data[0, 500] = "+"
    for rock in rocks:
        data[rock[1], rock[0]] = "#"
    data[-1, :] = "#"

    i = 1
    while data[0, 500] == "+":
        s = Sand(data)
        data, keep_going = s.flow(data)
        i += 1
    viz(data)
    print(f"Part 2: {i - 1}")
