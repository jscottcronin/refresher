def read_data(filename):
    with open(filename, "r") as f:
        data = [line.split(" ") for line in f.read().replace(" |", "").split("\n")]
    return data


def get_digit(line):
    m = {len(s): set(s) for s in line if len(s) in [2, 3, 4, 7]}
    out = ""
    # fmt: off
    for s in line[-4:]:
        match len(s), len(set(s) | m[4]), len(set(s) | m[2]):
            case 2, _, _: out += "1"
            case 3, _, _: out += "7"
            case 4, _, _: out += "4"
            case 7, _, _: out += "8"
            case 5, 7, _: out += "2"
            case 5, 6, 5: out += "3"
            case 5, 6, 6: out += "5"
            case 6, 6, _: out += "9"
            case 6, 7, 6: out += "0"
            case 6, 7, 7: out += "6"
    return int(out)


def part1(data):
    count = 0
    for line in data:
        count += sum([1 for s in line[-4:] if len(s) in {2, 3, 4, 7}])
    return count


def part2(data):
    return sum([get_digit(line) for line in data])


if __name__ == "__main__":
    fn = "day08/input.txt"
    data = read_data(fn)
    print(part1(data))
    print(part2(data))
