import numpy as np

from game2048.expectimax import board_to_move
from game2048.game import Game
from wmz.utils import onehot


class Supervisor:
    def __init__(self, score_to_win):
        self.board_size = 4
        self.score_to_win = score_to_win
        self.smart_agent_policy = board_to_move

    def push_R4(self, states, actions, b, m):
        for _ in range(4):
            states.append(b)
            actions.append(m)

            b = np.rot90(b, axes=(1, 2))
            m = (m + 1) % 4

    def push_D4(self, states, actions, board, move):
        self.push_R4(states, actions, board, move)
        self.push_R4(states, actions, np.flip(board), [0, 3, 2, 1][move])

    def imitate(self):
        game = Game(self.board_size, self.score_to_win)
        states = []
        actions = []
        while not game.end:
            s, a = onehot(game.board), board_to_move(game.board)
            self.push_R4(states, actions, s, a)
            game.move(a)

        return states, actions

    def instruct(self, cls_agent, **kwargs):
        game = Game(self.board_size, self.score_to_win)
        states = []
        actions = []
        naive_agent = cls_agent(game, display=None, **kwargs)
        while not game.end:
            s, a = onehot(game.board), board_to_move(game.board)
            self.push_D4(states, actions, s, a)
            game.move(naive_agent.step())

        return states, actions
