from collections import defaultdict
from functools import reduce
import statistics


def read_data(filename):
    with open(filename, "r") as f:
        return f.read().split("\n")


def soln(data):
    d = {
        ")": ("(", 3, 1),
        "]": ("[", 57, 2),
        "}": ("{", 1197, 3),
        ">": ("<", 25137, 4),
    }
    d_inv = {v: (k, e, s) for k, (v, e, s) in d.items()}
    errors = []
    scores = []
    for line in data:
        stack = []
        incomplete = True
        for c in line:
            if c not in d.keys():
                stack.append(c)
            else:
                v = stack.pop()
                if v != d[c][0]:
                    errors.append(d[c][1])
                    incomplete = False
        if len(stack) > 0 and incomplete:
            score = [0] + [d_inv[i][2] for i in stack[::-1]]
            scores.append(reduce(lambda x, y: 5 * x + y, score))
    return sum(errors), statistics.median(scores)


if __name__ == "__main__":
    fn = "day10/input.txt"
    data = read_data(fn)
    p1, p2 = soln(data)
    print(p1)
    print(p2)
