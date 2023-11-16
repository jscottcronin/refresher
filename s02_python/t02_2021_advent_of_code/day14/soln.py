from collections import Counter
from functools import lru_cache, reduce


def read_data(filename):
    with open(filename, "r") as f:
        template, ms = f.read().split("\n\n")
    data = {m.split(" -> ")[0]: m.split(" -> ")[1] for m in ms.split("\n")}
    return template, data


def part1(s, data):
    for _ in range(10):
        betweens = [data[s[i : i + 2]] for i in range(len(s) - 1)]
        s = "".join(list(map(str.__add__, s, betweens))) + s[-1]
    c = Counter(s)
    return max(c.values()) - min(c.values())


@lru_cache(maxsize=None)
def recurse(a, b, n=1):
    global data
    if n == 0:
        return Counter()
    x = data[a + b]
    return Counter(x) + recurse(a, x, n - 1) + recurse(x, b, n - 1)


def part2(template):
    cs = [recurse(i, j, n=40) for (i, j) in zip(template, template[1:])]
    cs = Counter(template) + reduce(lambda x, y: x + y, cs)
    return max(cs.values()) - min(cs.values())


def soln(s, data, n=10):
    counts = Counter(s)
    pairs = Counter(["".join(t) for t in zip(s, s[1:])])

    for _ in range(n):
        for (a, b), c in pairs.copy().items():
            x = data[a + b]
            pairs[a + b] -= c
            pairs[a + x] += c
            pairs[x + b] += c
            counts[x] += c
    return max(counts.values()) - min(counts.values())


if __name__ == "__main__":
    fn = "day14/input.txt"
    template, data = read_data(fn)

    # First attempt solutions
    print(part1(template, data))
    print(part2(template))

    # Generic solutions
    print(soln(template, data, n=10))
    print(soln(template, data, n=40))
