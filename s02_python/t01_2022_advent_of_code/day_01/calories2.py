import pandas as pd


data = {}

idx = 1
calories = 0

with open('data.txt') as f:
    for line in f:
        if line == '\n':
            data[idx] = calories
            calories = 0
            idx += 1
        else:
            calories += int(line.strip())

x = pd.Series(data).sort_values(ascending=False)[:3].sum()
print(x)