import re
from typing import List

import numpy as np


class Board:
    def __init__(self, board: List[List[str]]) -> None:
        self.board = np.array(board).astype(int)
        self.truth = np.zeros((5, 5), dtype=bool)
        self.score = None

    def update(self, call):
        if not self.score:
            self.truth[self.board == call] = True
            if np.all(self.truth, axis=0).any() or np.all(self.truth, axis=1).any():
                self.score = self.board[~self.truth].sum() * call
            return True


def read_data(filename):
    with open(filename, "r") as f:
        calls = [int(x) for x in f.readline().strip().split(",")]
        bs = [b.split("\n") for b in f.read().strip().split("\n\n")]
        bs = [[re.sub(r"\s+", " ", r.strip()).split(" ") for r in b] for b in bs]
    return calls, bs


def soln(calls, boards):
    scores = []
    while calls:
        call = calls.pop(0)
        for board in boards:
            updated = board.update(call)
            if updated and board.score:
                scores.append(board.score)
    return scores[0], scores[-1]


if __name__ == "__main__":
    fn = "day04/input.txt"
    calls, boards = read_data(fn)
    boards = [Board(board) for board in boards]
    print(soln(calls, boards))
