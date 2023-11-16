def read_data(filename):
    with open(filename, "r") as f:
        data = [int(x.strip()) for x in f.readlines()]
    return data


def part1(data):
    return sum(x < y for x, y in zip(data, data[1:]))


def part2(data):
    windows = [sum(data[i : i + 3]) for i in range(len(data) - 2)]
    return sum(x < y for x, y in zip(windows, windows[1:]))


if __name__ == "__main__":
    fn = "day01/input.txt"
    data = read_data(fn)
    print(part1(data))
    print(part2(data))
