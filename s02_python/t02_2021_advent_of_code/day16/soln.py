from collections import deque
from functools import reduce
from operator import *

hex2bit = {
    "0": "0000",
    "1": "0001",
    "2": "0010",
    "3": "0011",
    "4": "0100",
    "5": "0101",
    "6": "0110",
    "7": "0111",
    "8": "1000",
    "9": "1001",
    "A": "1010",
    "B": "1011",
    "C": "1100",
    "D": "1101",
    "E": "1110",
    "F": "1111",
}
version_sum = 0


def read_data(filename):
    with open(filename, "r") as f:
        data = f.read().strip()

    return data


def read_bits(q, n):
    return "".join(q.popleft() for i in range(n))


def bits_to_num(b):
    return int(b, 2)


def get_header(q):
    v = bits_to_num(read_bits(q, 3))
    type_id = bits_to_num(read_bits(q, 3))
    return v, type_id


def parse(q, n=-1):
    global version_sum
    # pop = lambda x: "".join([q.popleft() for i in range(x)])
    # h = lambda x: [k for k, v in hex2bit.items() if v.endswith(x) and k.isnumeric()][0]
    n -= 1
    if not any(q):
        return version_sum

    version, type_id = get_header(q)
    version_sum += version

    def get_subpackets():
        if type_id == 4:  # Literal Packet
            bits = read_bits(q, 5)
            while bits[-5] != "0":
                bits += read_bits(q, 5)
            bits = "".join([b for i, b in enumerate(bits) if i % 5 != 0])
            value = bits_to_num(bits)
            yield value
        else:  # operator type
            if read_bits(q, 1) == "1":
                subpackets = bits_to_num(read_bits(q, 11))
                for _ in range(subpackets):
                    yield parse(q, n=subpackets)
            else:
                subpackets_len = bits_to_num(read_bits(q, 15))
                q_len_delta = len(q) - subpackets_len
                while len(q) != q_len_delta:
                    yield parse(q)

        if n == 0:
            return

    f = [add, mul, min, max, lambda x, y: (int(x) << 4) | int(y), gt, lt, eq][type_id]
    return reduce(f, get_subpackets())


def part1(data):
    global version_sum
    q = deque()
    for c in data:
        q.extend(hex2bit[c])
    q = parse(q)
    return version_sum


def part2(data):
    q = deque()
    for c in data:
        q.extend(hex2bit[c])
    return parse(q)


if __name__ == "__main__":
    fn = "day16/input.txt"
    data = read_data(fn)
    # print(part1("8A004A801A8002F478"))
    print(part1(data))
    print(part2(data))
