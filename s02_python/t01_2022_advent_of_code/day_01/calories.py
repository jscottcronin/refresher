calories = []
elf_index = 1
elf_calories = 0

max_elf_index = 0
max_calories = 0

with open('data.txt') as f:
    for line in f:
        if line == '\n':
            calories.append(elf_calories)
            if elf_calories > max_calories:
                max_calories = elf_calories
                max_elf_index = elf_index
            elf_calories = 0
            elf_index += 1
        else:
            elf_calories += int(line.strip())

print(f'Elf Number {max_elf_index} carried {max_calories}')