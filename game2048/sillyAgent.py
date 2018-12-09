# -*- coding:utf-8 -*-

from game2048.agents import Agent
from game2048.game import _merge
import numpy as np


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


class SillyAgent(Agent):
    def __init__(self, game, display=None, depth=4, alpha=1.1, gamma=1.1):
        super().__init__(game, display)

        self.depth = depth
        self.alpha = alpha
        self.gamma = gamma

        self.actions = self._make_actions()
        self.weights = self._make_weights()

    def step(self):
        # enumerate all possible actions,
        # and record the possible scores.
        direction_and_score = []
        for action in self.actions:
            board = self.game.board
            score = 0.0
            factor = 1.0
            for direction in action:
                board = _try_move(board, direction)
                score += factor * self._get_weighted_score(board)
                factor *= self.alpha

                if (not list(zip(*np.where(board == 0)))):
                    break

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

    def _get_weighted_score(self, board):
        score = np.sum(self.weights * board)
        return score

    def _make_actions(self):
        actions = [[] * 4]
        for _ in range(self.depth):
            actions = [[*a, d] for a in actions for d in range(4)]
        return actions

    def _make_weights(self):
        weight = [[0, 1, 2, 3], [7, 6, 5, 4], [8, 9, 10, 11], [15, 14, 13, 12]]
        weight = [[self.gamma**c for c in r] for r in weight]
        weight = np.array(weight)
        return weight
