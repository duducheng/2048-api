#!/usr/bin/python
# -*- coding: utf-8 -*-

''' Help the user achieve a high score in a real game of 2048 by using a move searcher. '''

from __future__ import print_function
import ctypes
import time
import os
import numpy as np


# Enable multithreading?
MULTITHREAD = True

for suffix in ['so', 'dll', 'dylib']:
    dllfn = os.path.join(os.path.dirname(__file__), 'bin/2048.' + suffix)
    print("Loaded expectmax lib for 2048:", dllfn)
    if not os.path.isfile(dllfn):
        continue
    ailib = ctypes.CDLL(dllfn)
    break
else:
    print(
        "Couldn't find 2048 library bin/2048.{so,dll,dylib}! Make sure to build it first.")
    # exit()

ailib.init_tables()

ailib.find_best_move.argtypes = [ctypes.c_uint64]
ailib.score_toplevel_move.argtypes = [ctypes.c_uint64, ctypes.c_int]
ailib.score_toplevel_move.restype = ctypes.c_float


def to_c_board(m):
    board = 0
    i = 0
    for row in m:
        for c in row:
            board |= c << (4 * i)
            i += 1
    return board


def print_board(m):
    for row in m:
        for c in row:
            print('%8d' % c, end=' ')
        print()


def _to_val(c):
    if c == 0:
        return 0
    return 2**c


def to_val(m):
    return [[_to_val(c) for c in row] for row in m]


def _to_score(c):
    if c <= 1:
        return 0
    return (c - 1) * (2**c)


def to_score(m):
    return [[_to_score(c) for c in row] for row in m]


if MULTITHREAD:
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(4)

    def score_toplevel_move(args):
        return ailib.score_toplevel_move(*args)

    def find_best_move(m):
        board = to_c_board(m)

        scores = pool.map(score_toplevel_move, [
                          (board, move) for move in range(4)])
        bestmove, bestscore = max(enumerate(scores), key=lambda x: x[1])
        if bestscore == 0:
            return -1
        return bestmove
else:
    def find_best_move(m):
        board = to_c_board(m)
        move = ailib.find_best_move(board)
        return move


def m_to_move(m):
    '''
    expectmax: udlr
    mine: ldru
    '''
    move = find_best_move(m)
    return [3, 1, 0, 2][move]


def board_to_move(x):
    '''i.e, `m`'''
    arr = np.log2(x + (x == 0))
    ret = []
    for r in arr:
        ret.append([int(c) for c in r])
    move = m_to_move(ret)
    return move
