# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time


class StorableNet(nn.Module):
    def __init__(self):
        super(StorableNet, self).__init__()

    def forward(self, *inputs):
        raise NotImplementedError

    def load(self, path, use_gpu=False):
        if use_gpu:
            self.load_state_dict(
                torch.load(path, map_location=lambda s, l: s.cuda()))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


map_table = {2 ** i: i for i in range(1, 12)}
map_table[0] = 0


def onehot(board):
    onehot_board = np.zeros(shape=(12, 4, 4), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            onehot_board[map_table[board[r, c]], r, c] = 1
    return onehot_board


class MeanRecorder:
    def __init__(self):
        self.__record = {}

    def record(self, name, value):
        if self.__record.__contains__(name):
            self.__record[name].append(value)
        else:
            self.__record[name] = [value]

    def mean(self, name):
        if self.__record.__contains__(name):
            return sum(self.__record[name]) / len(self.__record[name])
        else:
            raise KeyError

    def clear(self, name=None):
        if name is not None:
            self.__record.pop(name)
        else:
            self.__record.clear()


class Timer:
    def __init__(self):
        self.t = time.time()

    def tik(self):
        tt = time.time()
        print(F"time(s): {tt - self.t}")
        self.t = tt
