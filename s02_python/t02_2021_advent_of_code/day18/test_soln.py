import os, pytest, sys

sys.path = [os.getcwd()] + sys.path

from day18.soln import _format_line, add_pair, add_pairs, N
from day18.soln_mocks import *


def test_format_line():
    input = "[[[[1,1],[2,2]],[3,3]],[4,4]]"
    actual = _format_line(input)
    expected = [N(1, 4), N(1, 4), N(2, 4), N(2, 4), N(3, 3), N(3, 3), N(4, 2), N(4, 2)]
    assert actual == expected


def test_add_pair():
    a, b = [N(5, 1)], [N(4, 1)]
    actual = add_pair(a, b)
    expected = [N(5, 2), N(4, 2)]
    assert actual == expected


def test_add_pairs():
    input = ["[1,1]", "[2,2]", "[3,3]", "[4,4]"]
    fmt = [_format_line(l) for l in input]
    actual = add_pairs(fmt)
    expected = [N(1, 4), N(1, 4), N(2, 4), N(2, 4), N(3, 3), N(3, 3), N(4, 2), N(4, 2)]
    assert actual == expected
