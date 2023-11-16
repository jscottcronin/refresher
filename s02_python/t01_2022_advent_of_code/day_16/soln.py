from collections import defaultdict
from dataclasses import dataclass, field
import functools
import itertools
import math
import re


def read_data(filename):
    pattern = r"Valve (\w+) .*=(\d+); .* valves? (.*)"
    return re.findall(pattern, open(f"day_16/{filename}").read())


@functools.lru_cache()
def search(t: int, u: str, vs: set) -> int:
    scores = [0]
    for v in vs:
        t_new = t - D[u, v] - 1
        if D[u, v] + 1 < t:
            score = F[v] * t_new + search(t_new, v, vs - {v})
            scores.append(score)
    return max(scores)


@functools.lru_cache()
def search_w_elephant(t: int, u: str, vs: set) -> int:
    scores = [0]
    for v in vs:
        t_new = t - D[u, v] - 1
        if D[u, v] + 1 < t:
            score = F[v] * t_new + search_w_elephant(t_new, v, vs - {v})
            scores.append(score)
    scores += [search(26, u="AA", vs=vs)]
    return max(scores)


def part2(t, vs, u="AA", e=False):
    scores = [0]
    for v in vs:
        if D[u, v] < t:
            t_new = t - D[u, v] - 1
            score = F[v] * t_new + part2(t_new, vs - {v}, v, e)
            scores.append(score)
    return max(scores + [part2(26, vs) if e else 0])


if __name__ == "__main__":
    start = "AA"
    data = read_data("data.txt")

    V = set()  # Vertices
    F = {}  # Flows where flow > 0
    D = defaultdict(lambda: 1000)  # distances

    for v, f, us in data:
        V.add(v)
        if f != "0":
            F[v] = int(f)
        for u in us.split(", "):
            D[v, u] = 1

    # floyd-warshall
    for k, i, j in itertools.product(V, V, V):
        D[i, j] = min(D[i, j], D[i, k] + D[k, j])

    # best_score = search(30, "AA", frozenset(F))
    # print(f"Part 1: {best_score}")

    elephant_score = part2(26, frozenset(F), "AA", True)
    print(f"Part 2: {elephant_score}")

    # print(search_w_elephant(26, u="AA", vs=frozenset(F)))
