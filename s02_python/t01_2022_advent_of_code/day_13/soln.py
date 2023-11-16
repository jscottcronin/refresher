import json


def comp(left, right):
    if isinstance(left, int) and isinstance(right, int):
        if left < right:
            return True
        elif left > right:
            return False
    elif isinstance(left, int) and isinstance(right, list):
        return comp([left], right)
    elif isinstance(left, list) and isinstance(right, int):
        return comp(left, [right])
    elif isinstance(left, list) and isinstance(right, list):
        for v in map(comp, left, right):
            if v is not None:
                return v
        return comp(len(left), len(right))


if __name__ == "__main__":
    with open("day_13/data.txt") as f:
        pairs = [
            tuple(map(json.loads, pair.split("\n")))
            for pair in "".join(f.readlines()).split("\n\n")
        ]
    part1 = 0
    for i, pair in enumerate(pairs):
        if comp(*pair) == True:
            part1 += i + 1
    print(f"Part 1: {part1}")

    packets = [item for pair in pairs for item in pair] + [[[2]], [[6]]]
    ordered_packets = []
    for i, packet in enumerate(packets):
        if i == 0:
            ordered_packets.append(packet)
        else:
            # print(packet, ordered_packets)
            order = [comp(packet, ordered_packet) for ordered_packet in ordered_packets]
            try:
                n = order.index(True)
            except ValueError:
                n = len(order)

            ordered_packets = ordered_packets[:n] + [packet] + ordered_packets[n:]
            # print(ordered_packets)
    marker1 = ordered_packets.index([[2]]) + 1
    marker2 = ordered_packets.index([[6]]) + 1
    print(f"Part 2: {marker1 * marker2}")
