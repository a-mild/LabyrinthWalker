from typing import Sequence, Tuple
from functools import lru_cache

import numpy as np
from scipy.signal import convolve2d

from src.puzzle import Puzzle
from src.kernels import KERNELS


DEFAULT = np.zeros((3, 3))
DEFAULT[0, 0] = 1

BORDER_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


#TODO: raise no solution as exception
#TODO method depth / breadth first search as enum
class LabyrinthPuzzle(Puzzle):
    goal = None

    def __init__(self, pos: np.array, cursor=(0, 0), direction=(0, 0)):
        self.pos = pos
        self.direction = direction
        self.cursor = cursor

    @classmethod
    def set_up(cls, board: np.array, start: Tuple[int, int], goal: Tuple[int, int]):
        board[start] = 1
        cls.goal = goal
        return cls(board, start)

    def isgoal(self):
        return self.pos[self.goal] == 1

    @lru_cache
    def distance_from_point(self, point: Sequence[int]):
        grid = np.indices(self.pos.shape)
        grid = grid.swapaxes(0, 2)
        vec = grid - point
        r2 = np.sum(np.power(vec, 2), axis=2)
        return r2
    
        
    def find_valid_fields(self):
        pos_dummy = np.zeros(self.pos.shape, dtype=bool)
        pos_dummy[self.cursor] = True
        valid_fields = convolve2d(pos_dummy, KERNELS[self.direction], mode="same")
        last_state = self.pos - pos_dummy
        borders = convolve2d(last_state, BORDER_KERNEL, mode="same")
        return valid_fields.astype(bool) & ~self.pos.astype(bool) & (borders < 1)

    def __iter__(self):
        valid_fields = self.find_valid_fields()
        valid_moves = np.argwhere(valid_fields)
        for move in valid_moves:
            new_pos = self.pos.copy()
            new_pos[tuple(move)] += 1
            direction = tuple(np.flip(move - self.cursor))
            yield LabyrinthPuzzle(pos=new_pos, cursor=tuple(move), direction=direction)

    def __repr__(self):
        return repr(self.pos)
