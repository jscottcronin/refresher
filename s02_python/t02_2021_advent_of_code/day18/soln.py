from collections import namedtuple


class N:
    def __init__(self, value, depth):
        self.value = value
        self.depth = depth

    def __repr__(self):
        return f"{self.value!r}+{self.depth}"

    def __add__(self, other):
        if isinstance(other, int):
            return N(self.value + other, self.depth)

    def __sub__(self, other):
        if isinstance(other, int):
            return N(self.value + other, self.depth)

    def __ge__(self, other):
        if isinstance(other, int):
            return self.value >= other

    def __bool__(self):
        return self.depth >= 5

    def __eq__(self, other):
        if isinstance(other, N):
            return self.value == other.value and self.depth == other.depth


def read_data(filename):
    with open(filename, "r") as f:
        data = f.read().strip().split("\n")
    return [_format_line(l) for l in data]


def _format_line(line):
    depth = 0
    fmt = []
    for c in line:
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
        elif c.isnumeric():
            fmt.append(N(int(c), depth))
    return fmt


def explode():
    """
    To explode a pair, the pair's left value is added to the first regular number to the
    left of the exploding pair (if any), and the pair's right value is added to the first
    regular number to the right of the exploding pair (if any). Exploding pairs will always
    consist of two regular numbers. Then, the entire exploding pair is replaced with the
    regular number 0.
    """
    pass


def reduce(pair):
    """
    To reduce a snailfish number, you must repeatedly do the first action in this list that
    applies to the snailfish number:
        If any pair is nested inside four pairs, the leftmost such pair explodes.
        If any regular number is 10 or greater, the leftmost such regular number splits.
    """
    while any([p.depth >= 5 for p in pair]):
        for i, p in enumerate(pair):
            if p.depth >= 5:
                try:
                    pair[i-1].value = p.value
                    pair[i+1].value = 




def change_depth(pair, n):
    return [N(i.value, i.depth + n) for i in pair]


def add_pair(a, b):
    pair = change_depth(a, 1) + change_depth(b, 1)
    
    
    return pair


def add_pairs(pairs):
    result = pairs.pop(0)
    for pair in pairs:
        result = add_pair(result, pair)
    return result


def part1(data):
    # xs = []
    # for item in data:
    pass


def part2(data):
    pass


if __name__ == "__main__":
    fn = "day18/input-dev.txt"
    data = read_data(fn)
    print(part1(data))
    print(part2(data))
