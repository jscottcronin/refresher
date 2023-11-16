from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np


def p06_zigzag(s: str, n: int) -> str:
    unit = list(range(n)) + list(range(n - 2, 0, -1))
    inds = (unit * (len(s) // len(unit) + 1))[: len(s)]
    c = "".join([x[0] for x in sorted(list(zip(s, inds)), key=lambda x: x[1])])
    return c


def p88_merge_sort(nums1: list[int], m: int, nums2: list[int], n: int):
    i = m - 1
    j = n - 1
    k = m + n - 1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            k -= 1
            i -= 1
        else:
            nums1[k] = nums2[j]
            k -= 1
            j -= 1
    if j >= 0:
        nums1[: k + 1] = nums2[: j + 1]
    return nums1


def p27_remove_element(nums: List[int], val: int) -> int:
    val_ind = 0
    for i, v in enumerate(nums):
        if v != val:
            nums[i] = nums[val_ind]
            nums[val_ind] = v
            val_ind += 1
    return val_ind


def p26_remove_dupes(nums: List[int]) -> int:
    dupes = set()
    dupes_ind = 0
    for i, v in enumerate(nums):
        if v not in dupes:
            nums[i] = nums[dupes_ind]
            nums[dupes_ind] = v
            dupes.add(v)
            dupes_ind += 1
    return dupes_ind


def p80_remove_dupes2(nums: List[int]):
    dupes = defaultdict(int)
    dupes_ind = 0
    for i, v in enumerate(nums):
        if dupes[v] < 2:
            nums[i] = nums[dupes_ind]
            nums[dupes_ind] = v
            dupes[v] += 1
            dupes_ind += 1
    return dupes_ind, nums


def p169_majority_element(nums: List[int]) -> int:
    c = Counter(nums)
    return c.most_common(1)[0][0]


def p189_rotate(nums: List[int], k: int) -> None:
    k = k % len(nums)
    nums[:] = nums[-k:] + nums[:-k]


def p12_int_to_roman(num: int) -> str:
    r = {
        1: "I",
        5: "V",
        10: "X",
        50: "L",
        100: "C",
        500: "D",
        1000: "M",
    }
    xs = [1000, 100, 10, 1]
    s = ""
    for x in xs:
        if num // x == 9:
            s += r[x] + r[x * 10]
        elif num // x in [5, 6, 7, 8]:
            s += r[5 * x] + r[x] * (num // x - 5)
        elif num // x == 4:
            s += r[x] + r[5 * x]
        else:
            s += r[x] * (num // x)
        num = num % x
    return s


def p55_can_jump(nums: List[int]):
    visited = {0}
    q = [0]
    while q:
        current_ind = q.pop(0)
        max_jump = nums[current_ind]
        for jump in range(max_jump + 1):
            trial_ind = current_ind + jump
            if trial_ind == len(nums) - 1:
                return True
            elif trial_ind not in visited and trial_ind <= len(nums):
                q.append(trial_ind)
                visited.add(trial_ind)
    return False


def p45_can_jump_2(nums: List[int]):
    if len(nums) == 1:
        return 0
    visited = defaultdict(int)
    visited[0] = 0
    q = [0]
    while q:
        current_ind = q.pop(0)
        max_jump = nums[current_ind]
        for jump in range(max_jump + 1):
            trial_ind = current_ind + jump
            if trial_ind not in visited and trial_ind < len(nums):
                q.append(trial_ind)
                visited[trial_ind] = visited[current_ind] + 1

                if trial_ind == len(nums) - 1:
                    break
    return visited[len(nums) - 1]


def p274_h_index(citations: List[int]) -> int:
    for i, v in enumerate(sorted(citations, reverse=True), 1):
        if v < i:
            return i - 1
    return len(citations)


def p68_text_justify(words: List[str], maxWidth: int) -> List[List[str]]:
    lines = []
    while words:
        use_words = []
        while (
            words
            and len(words[0] if not use_words else " ".join(use_words) + " " + words[0])
            <= maxWidth
        ):
            use_words.append(words.pop(0))
        lines.append(use_words)

    justifieds = []
    for i, line in enumerate(lines, 1):
        justified = ""
        slots = len(line) - 1
        spaces = maxWidth - sum([len(w) for w in line])
        if slots == 0:
            justified += line[0] + " " * spaces
            justifieds.append(justified)
            continue
        # This is for the last line ot be left justified
        if i == len(lines):
            justified += " ".join(line) + " " * (spaces - slots)
            justifieds.append(justified)
            continue
        addons = [" " * (spaces // slots)] * slots
        for i in range(spaces % slots):
            addons[i] += " "
        for i, addon in enumerate(addons):
            justified += line[i] + addon
        justified += line[-1]
        justifieds.append(justified)
    return justifieds


def p42_trap_rain_water(height: List[int]) -> int:
    water = 0
    height = [0] + height + [0]
    for i, h in enumerate(height[1:-1]):
        left = max(height[: i + 1])
        right = max(height[i + 2 :])
        w = min(left, right) - h
        if w > 0:
            water += w
    return water


def p135_candy(ratings: List[int]) -> int:
    candies = [1] * len(ratings)
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    for i in range(len(ratings) - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
    return sum(candies)


def p392_is_subsequence(s: str, t: str) -> bool:
    if len(s) == 0:
        return True
    if len(t) < len(s):
        return False
    sub = list(s)
    while sub:
        c = sub.pop(0)
        x = t.find(c)
        if x < 0:
            return False
        else:
            t = t[x + 1 :]
    return True


def p167_two_sum(numbers: List[int], target: int) -> List[int]:
    c = Counter()
    new_numbers = []
    for n in numbers:
        if c[n] < 2:
            new_numbers.append(n)
        c[n] += 1
    for i, n1 in enumerate(new_numbers):
        for j, n2 in enumerate(new_numbers[i + 1 :]):
            if n1 + n2 == target:
                a = numbers.index(n1) + 1
                b = a + numbers[a:].index(n2) + 1
                return [a, b]
    return [-1, -1]


if __name__ == "__main__":
    # numbers = [
    #     3,
    #     3,
    #     5,
    #     8,
    #     18,
    #     21,
    #     22,
    #     22,
    #     22,
    #     24,
    #     26,
    #     28,
    #     29,
    #     31,
    #     31,
    #     34,
    #     37,
    #     37,
    #     40,
    #     43,
    #     43,
    #     43,
    #     44,
    #     47,
    #     48,
    #     51,
    #     51,
    #     51,
    #     52,
    #     54,
    #     55,
    #     56,
    #     59,
    #     59,
    #     60,
    #     74,
    #     74,
    #     76,
    #     76,
    #     81,
    #     82,
    #     82,
    #     82,
    #     85,
    #     89,
    #     91,
    #     91,
    #     94,
    #     99,
    #     101,
    #     101,
    #     106,
    #     116,
    #     118,
    #     121,
    #     126,
    #     127,
    #     128,
    #     128,
    #     128,
    #     131,
    #     134,
    #     135,
    #     138,
    #     140,
    #     143,
    #     145,
    #     151,
    #     152,
    #     153,
    #     154,
    #     156,
    #     158,
    #     158,
    #     158,
    #     160,
    #     169,
    #     173,
    #     174,
    #     177,
    #     178,
    #     180,
    #     189,
    #     190,
    #     190,
    #     191,
    #     191,
    #     196,
    #     197,
    #     203,
    #     203,
    #     206,
    #     206,
    #     206,
    #     208,
    #     210,
    #     212,
    #     215,
    #     216,
    #     218,
    #     218,
    #     219,
    #     223,
    #     225,
    #     227,
    #     229,
    #     232,
    #     232,
    #     233,
    #     234,
    #     235,
    #     235,
    #     236,
    #     237,
    #     238,
    #     239,
    #     245,
    #     249,
    #     250,
    #     251,
    #     254,
    #     254,
    #     256,
    #     260,
    #     261,
    #     262,
    #     270,
    #     271,
    #     271,
    #     274,
    #     275,
    #     284,
    #     285,
    #     286,
    #     290,
    #     290,
    #     291,
    #     292,
    #     292,
    #     293,
    #     293,
    #     293,
    #     295,
    #     299,
    #     300,
    #     304,
    #     304,
    #     305,
    #     310,
    #     313,
    #     313,
    #     315,
    #     322,
    #     326,
    #     327,
    #     329,
    #     334,
    #     336,
    #     337,
    #     339,
    #     339,
    #     340,
    #     341,
    #     343,
    #     344,
    #     347,
    #     347,
    #     356,
    #     356,
    #     359,
    #     359,
    # ]
    numbers = [2, 7, 11, 15]
    print(p167_two_sum(numbers, 9))
