from collections import defaultdict, deque
import string

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(filename):
    str_to_int = {c: i + 1 for i, c in enumerate(string.ascii_lowercase)}
    str_to_int.update({"S": 1, "E": 26})

    data = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if "S" in line.strip():
                start = (i, line.index("S"))
            if "E" in line.strip():
                end = (i, line.index("E"))
            data.append([str_to_int[x] for x in list(line.strip())])

    return np.array(data), start, end


def is_valid_move(arr, node, new_node, visited):
    # must not be a place we've visited
    not_visited = new_node not in visited

    # must be inside the borders of the matrix
    within_borders_x = 0 <= new_node[0] < arr.shape[0]
    within_borders_y = 0 <= new_node[1] < arr.shape[1]

    # must not more than 1 level higher
    is_steady_climb = False
    if within_borders_x and within_borders_y:
        is_steady_climb = arr[new_node] - arr[node] <= 1

    valid = is_steady_climb & not_visited & within_borders_x & within_borders_y
    return valid


if __name__ == "__main__":
    data, start, end = load_data("day_12/data.txt")

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    starts = list(zip(*np.where(data == 1)))
    results = []

    for start in tqdm(starts):
        q = deque([(start, [])])
        visited = {start}
        found_end = False
        while q:
            node, path = q.popleft()
            if node == end:
                found_end = True
                break
            for move in moves:
                new_node = (node[0] + move[0], node[1] + move[1])
                if is_valid_move(data, node, new_node, visited):
                    visited.add(new_node)
                    new_path = path + [(new_node)]
                    q.append((new_node, new_path))
        if found_end:
            results.append((len(path), start))
    print(pd.DataFrame(results, columns=["dist", "start"]).sort_values("dist"))
