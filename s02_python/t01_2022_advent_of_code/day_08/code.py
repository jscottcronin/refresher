import numpy as np
import pandas as pd

df = pd.read_csv('data.txt', header=None)
data = np.array(df[0].map(lambda x: [int(c) for c in str(x)]).to_list())

c = (data.shape[0] * data.shape[1]) - ((data.shape[0] - 2) * (data.shape[1] - 2))
for i in range(1, data.shape[0]-1):
    for j in range(1, data.shape[1]-1):
        left = max(data[i, :j]) < data[i, j]
        right = max(data[i, j+1:]) < data[i, j]
        up = max(data[:i, j]) < data[i, j]
        down = max(data[i+1:, j]) < data[i, j]
        if left or right or up or down:
            c += 1
print('Part 1 solution:', c)


def get_view_distance(site, trees):
    vd = 0
    if not trees:
        return vd
    for tree in trees:
        vd += 1
        if tree >= site:
            break
    return vd

max_view = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # print(i, j)
        left = get_view_distance(data[i, j], data[i, :j].tolist()[::-1])
        right = get_view_distance(data[i, j], data[i, j+1:].tolist())
        up = get_view_distance(data[i, j], data[:i, j].tolist()[::-1])
        down = get_view_distance(data[i, j], data[i+1:, j].tolist())
        score = left * right * up * down
        # print(left, right, up, down, score)
        if score > max_view:
            max_view = score
print('Part 2 solution: ', max_view)
