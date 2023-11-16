from collections import Counter
import string

def split_string(s):
    return s[:len(s)//2], s[len(s)//2:]

def find_duplicate(a, b):
    a = set(Counter(a))
    b = set(Counter(b))
    i = a.intersection(b)
    return set(i).pop()

def get_priority(x):
    letters = list(string.ascii_lowercase + string.ascii_uppercase)
    numbers = list(range(1,53))
    priorities = dict(zip(letters, numbers))
    return priorities[x]

priority_score = 0
with open('data.txt') as f:
    for line in f:
        a, b = split_string(line.strip())
        x = find_duplicate(a, b)
        s = get_priority(x)
        priority_score += s

    
print(priority_score)

