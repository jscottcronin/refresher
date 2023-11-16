if __name__ == "__main__":
    with open("day_01/data.txt") as f:
        elves = [elf.split("\n") for elf in f.read().split("\n\n")]
        cals = [sum(map(int, elf)) for elf in elves]

    print(f"Part 1: {max(cals)}")
    print(f"Part 2: {sum(sorted(cals)[-3:])}")
