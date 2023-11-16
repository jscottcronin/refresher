from functools import lru_cache
import numpy as np


def read_data(filename):
    with open(filename, "r") as f:
        data = np.array(f.read().strip().split(","), dtype=int)
    return data


def soln(data, part2=False):
    min, max = data.min(), data.max()
    fuel_min = 1e12
    fuel_delta = np.arange(max + 1).cumsum()
    for pos in range(min, max + 1):
        deltas = abs(data - pos)
        if not part2:
            fuel = deltas.sum()
        else:
            fuel = fuel_delta[deltas].sum()
        if fuel < fuel_min:
            fuel_min = fuel
    return fuel_min


if __name__ == "__main__":
    fn = "day07/input.txt"
    data = read_data(fn)
    print(soln(data))
    print(soln(data, part2=True))
