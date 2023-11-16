from collections import defaultdict

cycle = 0
x = 1
signal = defaultdict(int)

with open("Day 10/data.txt") as f:
    for line in f:
        line = line.strip().split(" ")
        cs = len(line)
        if len(line) > 1:
            n = int(line[1])

        for c in range(cs):
            cycle += 1
            signal[cycle] = x
            if c == 1:
                x += n

watcher = {20, 60, 100, 140, 180, 220}
score = sum([k * v for k, v in signal.items() if k in watcher])
text = ["#" if abs((k - 1) % 40 - v) <= 1 else " " for k, v in signal.items()]
print("\n".join(["".join(text[x : x + 40]) for x in range(0, len(text), 40)]))
