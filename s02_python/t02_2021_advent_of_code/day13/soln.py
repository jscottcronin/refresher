import numpy as np


def read_data(filename):
    with open(filename, "r") as f:
        data, folds = f.read().split("\n\n")
    data = [tuple(map(int, s.split(","))) for s in data.split("\n")]

    x_max = sorted(data, key=lambda t: -t[0])[0][0]
    y_max = sorted(data, key=lambda t: -t[1])[0][1]
    arr = np.zeros((x_max + 1, y_max + 1))
    for point in data:
        arr[point] = 1
    folds = [(fold[11:12], int(fold[13:])) for fold in folds.split("\n")]
    return arr, folds


def soln(arr, folds, part_two=False):
    for fold, value in folds:
        if fold == "y":
            a = arr[:, :value]
            b = (arr[:, value + 1 :])[:, ::-1]
            if a.shape[1] > b.shape[1]:
                a[:, a.shape[1] - b.shape[1] :] += b
                arr = a
            elif a.shape[1] < b.shape[1]:
                b[:, b.shape[1] - a.shape[1] :] += a
                arr = b
            else:
                arr = a + b
        if fold == "x":
            a = arr[:value, :]
            b = (arr[value + 1 :, :])[::-1, :]
            if a.shape[0] > b.shape[0]:
                a[a.shape[0] - b.shape[0] :, :] += b
                arr = a
            elif a.shape[0] < b.shape[0]:
                b[b.shape[0] - a.shape[0] :, :] += a
                arr = b
            else:
                arr = a + b

        if not part_two:
            return (arr > 0).sum().sum()

    cs = np.full(arr.shape, " ", dtype=str)
    cs[arr > 0] = "#"
    return "\n".join(["".join(l) for l in cs.T.tolist()])


if __name__ == "__main__":
    fn = "day13/input.txt"
    data, folds = read_data(fn)
    print(soln(data, folds))
    print(soln(data, folds, part_two=True))
