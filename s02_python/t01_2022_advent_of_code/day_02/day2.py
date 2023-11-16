def game(a, b):
    a = {'A': 'Rock', 'B': 'Paper', 'C': 'Scissors'}[a]
    b = {'X': 'Rock', 'Y': 'Paper', 'Z': 'Scissors'}[b]

    if a == b:
        return 3
    elif a == 'Rock' and b == 'Paper':
        return 6
    elif a == 'Paper' and b == 'Scissors':
        return 6
    elif a == 'Scissors' and b == 'Rock':
        return 6
    else:
        return 0


def points(b):
    return {'X': 1, 'Y': 2, 'Z': 3}[b]


score = 0
with open('data.txt') as f:
    for line in f:
        a = line[0:1]
        b = line[2:3]
        score += game(a,b) + points(b)

print(score)

