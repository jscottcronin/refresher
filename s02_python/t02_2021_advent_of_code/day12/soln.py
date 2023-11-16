from collections import defaultdict


def read_data(filename):
    data = defaultdict(list)
    with open(filename, "r") as f:
        for line in f:
            a, b = line.strip().split("-")
            data[a].append(b)
            data[b].append(a)
    return data


def soln(graph, start, end, path, visits, part_two=False):
    if start == end:
        return 1

    paths = 0
    for neighbor in graph[start]:
        if not part_two:
            fltr = neighbor.islower() and visits[neighbor] > 0
        else:
            fltr = (
                neighbor.islower()
                and visits[neighbor] > 0
                and any([k.islower() and v == 2 for k, v in visits.items()])
            )
        if fltr or neighbor == "start":
            continue
        visits[neighbor] += 1
        paths += soln(graph, neighbor, end, path + [neighbor], visits, part_two)
        visits[neighbor] -= 1
    return paths


if __name__ == "__main__":
    fn = "day12/input.txt"
    data = read_data(fn)

    path = ["start"]
    visits = defaultdict(lambda: 0, {"start": 1})
    print(soln(data, "start", "end", path, visits, part_two=False))
    print(soln(data, "start", "end", path, visits, part_two=True))
