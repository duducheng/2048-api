import numpy as np
import torch
from torchvision import transforms
import os
from Net import myNet
from Net import testNet
class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display
    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction

class MyAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError("`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        modelPath = os.getcwd() + '/model/model.pth'
        model=myNet()
        model.load_state_dict(torch.load(modelPath))
        self.search_func = model

    def pre_process(self):
        board = self.game.board
        board[board==0] = 1
        board = np.log2(board).flatten()
        board = board.reshape((4, 4))
        board = board[:, :, np.newaxis]
        board = transforms.ToTensor()(board)
        board = torch.unsqueeze(board, dim=0).float()
        return board

    def step(self):
        input = self.pre_process()
        output = self.search_func(input)
        max_idx = np.where(output==torch.max(output))[1]
        counts = np.bincount(max_idx)
        direction = np.argmax(counts)
        return direction
        
class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
