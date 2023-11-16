import functools
import math
from collections import Counter
from tqdm import tqdm


class Monkey:
    def __init__(self, starting_items, operation, test, result_monkey):
        self.items = starting_items
        self.operation = operation
        self.test = test
        self.test_true = result_monkey[0]
        self.test_false = result_monkey[1]

    def throw_item_to_monkey(self, item, base, divide_by_three=True):
        if divide_by_three:
            worry = math.floor(self.operation(item) / 3)
        else:
            worry = self.operation(item)

        worry = worry % base
        test = worry % self.test == 0
        to_monkey = self.test_true if test else self.test_false
        return to_monkey, worry


if __name__ == "__main__":
    # Small data
    # starting_items = [[79, 98], [54, 65, 75, 74], [79, 60, 97], [74]]
    # operations = [lambda x: x * 19, lambda x: x + 6, lambda x: x * x, lambda x: x + 3]
    # tests = [23, 19, 13, 17]
    # results = [(2, 3), (2, 0), (1, 3), (0, 1)]

    # Big data
    starting_items = [
        [56, 52, 58, 96, 70, 75, 72],
        [75, 58, 86, 80, 55, 81],
        [73, 68, 73, 90],
        [72, 89, 55, 51, 59],
        [76, 76, 91],
        [88],
        [64, 63, 56, 50, 77, 55, 55, 86],
        [79, 58],
    ]
    operations = [
        lambda x: x * 17,
        lambda x: x + 7,
        lambda x: x * x,
        lambda x: x + 1,
        lambda x: x * 3,
        lambda x: x + 4,
        lambda x: x + 8,
        lambda x: x + 6,
    ]
    tests = [
        11,
        3,
        5,
        7,
        19,
        2,
        13,
        17,
    ]
    results = [
        (2, 3),
        (6, 5),
        (1, 7),
        (2, 7),
        (0, 3),
        (6, 4),
        (4, 0),
        (1, 5),
    ]

    inspection_counter = Counter()
    base = math.lcm(*tests)

    monkeys = [
        Monkey(*args) for args in zip(starting_items, operations, tests, results)
    ]
    for round_of_monkeys in tqdm(range(10000)):
        for i, monkey in enumerate(monkeys):
            for item in range(len(monkey.items)):
                inspection_counter[i] += 1
                old_worry = monkey.items.pop(0)
                to_monkey, worry = monkey.throw_item_to_monkey(
                    old_worry, base, divide_by_three=False
                )
                monkeys[to_monkey].items.append(worry)
                # print(f"Monkey {i}: {old_worry}->{worry} -> Monkey: {to_monkey}")
        # for monkey in monkeys:
        #     print(monkey.items)

    score = functools.reduce(
        lambda x, y: x[1] * y[1], inspection_counter.most_common(2)
    )
    print(inspection_counter)
    print(score)
