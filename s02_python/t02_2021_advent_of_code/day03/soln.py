from collections import Counter
import copy


def read_data(filename):
    with open(filename, "r") as f:
        data = [s.strip() for s in f.readlines()]
    return data


def get_rating(data, o2=True):
    d = copy.deepcopy(data)
    fltr = ""
    for i in range(len(d[0])):
        c = Counter([s[i] for s in d])
        if o2:
            fltr += str(int(c["1"] >= c["0"]))
        else:
            fltr += str(int(c["1"] < c["0"]))
        while not all([s.startswith(fltr) for s in d]):
            if len(d) == 1:
                break
            del d[[s.startswith(fltr) for s in d].index(False)]
    return d[0]


def part1(data):
    transpose = [[s[i] for s in data] for i in range(len(data[0]))]
    binary = "".join([Counter(x).most_common(1)[0][0] for x in transpose])
    gamma = int(binary, 2)
    epsilon = int("".join(["0" if c == "1" else "1" for c in binary]), 2)
    return gamma * epsilon


def part2(data):
    o2 = get_rating(data, o2=True)
    co2 = get_rating(data, o2=False)
    return int(o2, 2) * int(co2, 2)


if __name__ == "__main__":
    fn = "day03/input.txt"
    data = read_data(fn)
    print(part1(data))
    print(part2(data))
