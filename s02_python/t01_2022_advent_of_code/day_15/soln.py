from collections import Counter
from dataclasses import dataclass, field
import re

import numpy as np
from tqdm import tqdm


@dataclass
class Sensor:
    sensor: (int, int)
    beacon: (int, int)
    dist: int = field(init=False)
    invalid: set((int, int)) = field(init=False)

    min_x: int = field(default=0)
    max_x: int = field(default=0)
    min_y: int = field(default=0)
    max_y: int = field(default=0)

    def __post_init__(self):
        x_delta = abs(self.sensor[0] - self.beacon[0])
        y_delta = abs(self.sensor[1] - self.beacon[1])
        self.dist = x_delta + y_delta
        self.invalid = set([self.sensor, self.beacon])

        x, y = self.sensor[0], self.sensor[1]
        for dx in range(self.dist + 1):
            for dy in range(self.dist + 1):
                if dx + dy <= self.dist:
                    self.invalid.add((x + dx, y + dy))
                    self.invalid.add((x + dx, y - dy))
                    self.invalid.add((x - dx, y + dy))
                    self.invalid.add((x - dx, y - dy))
                    if x - dx < self.min_x:
                        self.min_x = x - dx
                    elif x + dx > self.max_x:
                        self.max_x = x + dx
                    if y - dy < self.min_y:
                        self.min_y = y - dy
                    elif y + dy > self.max_y:
                        self.max_y = y + dy


def viz(arr):
    # fmt: off
    print()
    print("\n".join(["".join(list(a)) for a in view]))
    print()


# if __name__ == "__main__":
with open("day_15/data.txt") as f:
    sensors = []
    pattern = r"x=(-?\d+),\s*y=(-?\d+)"
    for line in tqdm(f):
        matches = [tuple(map(int, m)) for m in re.findall(pattern, line)]
        sensors.append(Sensor(*matches))

min_x = min([s.min_x for s in sensors])
max_x = max([s.max_x for s in sensors])
min_y = min([s.min_y for s in sensors])
max_y = max([s.max_y for s in sensors])

# fmt: off
v_min_x = abs(min_x - min([s.sensor[0] for s in sensors] + [s.beacon[0] for s in sensors]))
v_min_y = abs(min_y - min([s.sensor[1] for s in sensors] + [s.beacon[1] for s in sensors]))
v_max_x = abs(min_x - max([s.sensor[0] for s in sensors] + [s.beacon[0] for s in sensors]))
v_max_y = abs(min_y - max([s.sensor[1] for s in sensors] + [s.beacon[1] for s in sensors]))

arr = np.full((max_y - min_y + 1, max_x - min_x + 1), ".")
for s in sensors:
    # if s.sensor == (8,7):
    for i in s.invalid:
        arr[i[1] - min_y, i[0] - min_x] = "#"
for s in sensors:
    if s.sensor[0] - min_x < 0:
        print(s.sensor)
    arr[s.sensor[1] - min_y, s.sensor[0] - min_x] = "S"
    arr[s.beacon[1] - min_y, s.beacon[0] - min_x] = "B"

row = 2000000
view = arr[v_min_y : v_max_y + 1, v_min_x : v_max_x + 1]
# viz(view)
c = Counter(''.join(list(view[row, :])))
print(c)
