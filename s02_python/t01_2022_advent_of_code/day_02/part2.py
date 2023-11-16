def make_my_choice(a, result):
    if result == 'Y':
        return a
    elif result == 'Z':
        if a == 'A':
            return 'B'
        elif a == 'B':
            return 'C'
        else:
            return 'A'
    else:
        if a == 'A':
            return 'C'
        elif a == 'B':
            return 'A'
        else:
            return 'B'


def game(a, b):
    a = {'A': 'Rock', 'B': 'Paper', 'C': 'Scissors'}[a]
    b = {'A': 'Rock', 'B': 'Paper', 'C': 'Scissors'}[b]
    # b = {'A': 'Rock', 'B': 'Paper', 'C': 'Scissors'}[b]

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
    return {'A': 1, 'B': 2, 'C': 3}[b]


score = 0
with open('data.txt') as f:
    for line in f:
        a = line[0:1]
        result = line[2:3]
        b = make_my_choice(a, result)
        score += game(a,b) + points(b)

print(score)

