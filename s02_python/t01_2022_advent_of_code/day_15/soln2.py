from collections import Counter
from dataclasses import dataclass, field
import re

import numpy as np
from tqdm import tqdm


@dataclass
class Array:
    sensors: list
    x_offset: int = field(init=False)
    y_offset: int = field(init=False)
    arr: np.array = field(init=False)

    def __post_init__(self):
        # fmt: off
        xs = [s.sensor[0] - s.dist for s in self.sensors] + [s.sensor[0] + s.dist for s in self.sensors]
        ys = [s.sensor[1] - s.dist for s in self.sensors] + [s.sensor[1] + s.dist for s in self.sensors]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        print(x_min, x_max, y_min, y_max)
        self.x_offset = -x_min
        self.y_offset = -y_min
        self.arr = np.zeros(x_max - x_min + 1)


@dataclass
class Sensor:
    sensor: (int, int)
    beacon: (int, int)
    perim: set = field(default_factory=set)
    dist: int = field(init=False)

    def __post_init__(self):
        x_delta = abs(self.sensor[0] - self.beacon[0])
        y_delta = abs(self.sensor[1] - self.beacon[1])
        self.dist = x_delta + y_delta

        # get perimeter points
        x, y = self.sensor
        p = self.dist + 1

        ur = set(zip(np.arange(x, x + p + 1, 1), np.arange(y - p, y + 1, 1)))
        lr = set(zip(np.arange(x, x + p + 1, 1), np.arange(y + p, y - 1, -1)))
        ul = set(zip(np.arange(x - p, x + 1, 1), np.arange(y, y + p + 1, 1)))
        ll = set(zip(np.arange(x - p, x + 1, 1), np.arange(y, y - p - 1, -1)))

        self.perim = self.perim | ur | lr | ul | ll


def man_dist(t1, t2):
    return abs(t2[0] - t1[0]) + abs(t2[1] - t1[1])


def not_within_sensor(point, sensor):
    dist = man_dist(point, sensor.sensor)
    return dist > sensor.dist


def part1(sensors, line):
    arr = Array(sensors)
    for sensor in arr.sensors:
        sx, sy = sensor.sensor
        excess = sensor.dist - abs(line - sy)
        if excess >= 0:
            x_min = sx - excess + arr.x_offset
            x_max = sx + excess + 1 + arr.x_offset
            arr.arr[x_min:x_max] = 1

    for sensor in arr.sensors:
        if sensor.sensor[1] == line:
            arr.arr[sensor.sensor[0] + arr.x_offset] = -1
        if sensor.beacon[1] == line:
            arr.arr[sensor.beacon[0] + arr.x_offset] = -2

    return sum(arr.arr == 1)


def part2(sensors, bounds):
    results = set()
    points = set()
    for sensor in tqdm(sensors):
        for p in sensor.perim:
            if 0 <= p[0] <= bounds and 0 <= p[1] <= bounds:
                points.add(p)

    for point in tqdm(points):
        if all([not_within_sensor(point, sensor) for sensor in sensors]):
            results.add(point)
    result = results[0]
    tuning = result[0] * 4000000 + result[1]
    return tuning


if __name__ == "__main__":
    with open("day_15/data.txt") as f:
        sensors = []
        pattern = r"x=(-?\d+),\s*y=(-?\d+)"
        for line in f:
            matches = [tuple(map(int, m)) for m in re.findall(pattern, line)]
            sensors.append(Sensor(*matches))
    print(f"Part 1: {part1(sensors, 2000000)}")
    print(f"Part 2: {part2(sensors, 4000000)}")
