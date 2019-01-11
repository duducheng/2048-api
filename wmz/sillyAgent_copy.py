# -*- coding:utf-8 -*-

from game2048.agents import Agent
from game2048.game import _merge
import numpy as np
from wmz.utils import onehot
from wmz.SimpleNet import SimpleNet

USE_GPU = True


class SillyAgent_copy(Agent):
    def __init__(self, game, display=None, model=None, model_path=None):
        super(SillyAgent_copy, self).__init__(game, display)

        if model is None:
            self.model = SimpleNet()
        else:
            self.model = model

        if USE_GPU:
            self.model.cuda()

        if model_path is not None:
            self.model.load(model_path, USE_GPU)

    def step(self):
        self.model.eval()

        # state = onehot(self.game.board)
        # direction = self.model.predict_direction(state, use_gpu=USE_GPU)

        onehot_board = onehot(self.game.board)
        ds = np.zeros(shape=(4, ))

        for i in range(4):
            state = np.rot90(onehot_board, k=i, axes=(1, 2)).copy()
            d = self.model.predict_direction(state, use_gpu=USE_GPU)
            ds[(d - i) % 4] += 1

        # onehot_board = np.flip(onehot_board,axis=1)
        # for i in range(4):
        #     state = np.rot90(onehot_board, k=i, axes=(1, 2)).copy()
        #     d = self.model.predict_direction(state, use_gpu=USE_GPU)
        #     ds[[0,3,2,1][(d - i) % 4]] += 1

        return int(np.argmax(ds))