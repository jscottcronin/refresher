import pytest
from day_13.soln import comp


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1], [1], None),
        (1, 1, None),
        (1, 2, True),
        (2, 1, False),
        ([2, 3, 4], 4, True),
        ([7, 7], [7], False),
        ([7], [7, 7], True),
        ([1, 1, 3, 1, 1], [1, 1, 5, 1, 1], True),
        ([[1], [2, 3, 4]], [[1], 4], True),
        ([9], [[8, 7, 6]], False),
        ([[4, 4], 4, 4], [[4, 4], 4, 4, 4], True),
        ([7, 7, 7, 7], [7, 7, 7], False),
        ([], [3], True),
        ([[[]]], [[]], False),
        (
            [1, [2, [3, [4, [5, 6, 7]]]], 8, 9],
            [1, [2, [3, [4, [5, 6, 0]]]], 8, 9],
            False,
        ),
        ([[0, 0], 2], [[0, 0], 1], False),
        ([[8, [[7]]]], [[[[8]]]], False),
        ([[1, 2], 4], [[1], 5, 5], False),
        ([[1, 2], 4], [[[3]], 5, 5], True),
        ([1, 2], [[3]], True),
    ],
)
def test_comp(a, b, expected):
    actual = comp(a, b)
    assert actual == expected
