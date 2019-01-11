# -*- coding:utf-8 -*-

from game2048.agents import Agent
from game2048.game import _merge
import numpy as np

DEPTH = 4
ALPHA = 1.1
GAMMA = 1.1

ACTIONS = [[] * 4]
for _ in range(DEPTH):
    ACTIONS = [[*a, d] for a in ACTIONS for d in range(4)]

WEIGHTS = [[0, 1, 2, 3], [7, 6, 5, 4], [8, 9, 10, 11], [15, 14, 13, 12]]
WEIGHTS = [[GAMMA**c for c in r] for r in WEIGHTS]
WEIGHTS = np.array(WEIGHTS)

get_weighted_score = lambda board: np.sum(WEIGHTS * board)


def _try_move(board, direction):
    '''
    direction:
        0: left
        1: down
        2: right
        3: up
    '''
    # treat all direction as left (by rotation)
    board_to_left = np.rot90(board, -direction)
    for row in range(board.shape[0]):
        core = _merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0

    # rotation to the original
    board_to_left = np.rot90(board_to_left, direction)
    return board_to_left


def board_to_move(game_board):
    # enumerate all possible actions,
    # and record the possible scores.
    direction_and_score = []
    for action in ACTIONS:
        board = np.array(game_board)
        score = 0.0
        factor = 1.0
        for direction in action:
            board = _try_move(board, direction)
            if (not list(zip(*np.where(board == 0)))):
                break

            score += factor * get_weighted_score(board)
            factor *= ALPHA

        direction_and_score.append([action[0], score])

    # choose the action that get the max score
    directions = []
    max_score = 0.0
    for d, s in direction_and_score:
        if s > max_score:
            max_score = s
            directions = [d]
        elif s == max_score:
            directions.append(d)
    greed_direction = np.random.choice(directions)

    return int(greed_direction)


class SillyAgent(Agent):
    def __init__(self, game, display=None):
        super().__init__(game, display)

    def step(self):
        return board_to_move(self.game.board)
