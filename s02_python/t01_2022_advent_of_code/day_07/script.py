def update_dict(d, keys, values):
    if len(keys) == 1:
        d[keys[0]] = values
        return
    return update_dict(d[keys[0]], keys[1:], values)


fs = {}
path = []
content = {}
update = False

with open('Day 7/data.txt') as f:
    for line in f:
        line = line.strip()

        if line.startswith('$') and update:
            update_dict(fs, path, content)
            update = False

        if line.startswith('$ cd ..'):
            path.pop()
        elif line.startswith('$ cd '):
            path.append(line.split(' ')[-1])
        elif line == '$ ls':
            update = True
            content = {}
        else:
            x, y = line.split(' ')
            if x == 'dir':
                content[y] = {}
            else:
                content[y] = int(x)
    update_dict(fs, path, content)
    
    results = {}
    
    def sum_file_sizes(fs, results, local_dir=''):
        local_sum = 0
        for k, v in fs.items():
            local_dir = '/'.join([local_dir, k]).replace('//', '/')
            if isinstance(v, dict):
                sub_sum = sum_file_sizes(fs[k], results, local_dir)
                results[local_dir] = sub_sum
                local_sum += sub_sum
            else:
                local_sum += v
        return local_sum
    
    sum_file_sizes(fs, results)
    answer = 0
    for k, v in results.items():
        if v <= 100000:
            answer += v
    print(f'Part A Answer: {answer}')


    total = results['/']
    min_delete = total - 40000000
    options = []
    for k, v in results.items():
        if v >= min_delete:
            options.append(v)
    print(f'Part B Answer: {sorted(options)[0]}')
    print(min_delete)
    print(results)
    