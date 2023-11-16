import re


def read_data(filename):
    with open(filename, "r") as f:
        pattern = r"(\w+) (\d+)"
        matches = [(d, int(x)) for d, x in re.findall(pattern, f.read())]
    return matches


def soln(data):
    position, depth, aim = 0, 0, 0
    for direction, value in data:
        match direction:
            case "down":
                aim += value
            case "up":
                aim -= value
            case "forward":
                position += value
                depth += value * aim
    return (position * aim), (position * depth)


if __name__ == "__main__":
    fn = "day02/input.txt"
    data = read_data(fn)
    part1, part2 = soln(data)
    print(part1, part2)
