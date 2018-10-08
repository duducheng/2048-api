'''A numpy-based 2048 game core implementation.'''

import numpy as np


class Game:

    def __init__(self, size=4, score_to_win=None, rate_2=0.5, random=False, enable_rewrite_board=False):
        '''

        :param size: the size of the board
        :param score_to_win: the terminate score to indicate `win`
        :param rate_2: the probability of the next element to be 2 (otherwise 4)
        :param random: a random initialized board (a harder mode)
        '''
        self.size = size
        if score_to_win is None:
            score_to_win = np.inf
        self.score_to_win = score_to_win
        self.__rate_2 = rate_2
        if random:
            self.__board = \
                2 ** np.random.randint(1, 10, size=(self.size, self.size))
            self.__end = False
        else:
            self.__board = np.zeros((self.size, self.size))
            # initilize the board (with 2 entries)
            self._maybe_new_entry()
            self._maybe_new_entry()
        self.enable_rewrite_board = enable_rewrite_board
        assert not self.end

    def move(self, direction):
        '''
        direction:
            0: left
            1: down
            2: right
            3: up
        '''
        # treat all direction as left (by rotation)
        board_to_left = np.rot90(self.board, -direction)
        for row in range(self.size):
            core = _merge(board_to_left[row])
            board_to_left[row, :len(core)] = core
            board_to_left[row, len(core):] = 0

        # rotation to the original
        self.__board = np.rot90(board_to_left, direction)
        self._maybe_new_entry()

    def __str__(self):
        board = "State:"
        for row in self.board:
            board += ('\t' + '{:8d}' *
                      self.size + '\n').format(*map(int, row))
        board += "Score: {0:d}".format(self.score)
        return board

    @property
    def board(self):
        '''`NOTE`: Setting board by indexing,
        i.e. board[1,3]=2, will not raise error.'''
        return self.__board.copy()

    @board.setter
    def board(self, x):
        if self.enable_rewrite_board:
            assert self.__board.shape == x.shape
            self.__board = x.astype(self.__board.dtype)
        else:
            print("Disable to rewrite `board` manually.")

    @property
    def score(self):
        return int(self.board.max())

    @property
    def end(self):
        '''
        0: continue
        1: lose
        2: win
        '''
        if self.score >= self.score_to_win:
            return 2
        elif self.__end:
            return 1
        else:
            return 0

    def _maybe_new_entry(self):
        '''maybe set a new entry 2 / 4 according to `rate_2`'''
        where_empty = self._where_empty()
        if where_empty:
            selected = where_empty[np.random.randint(0, len(where_empty))]
            self.__board[selected] = \
                2 if np.random.random() < self.__rate_2 else 4
            self.__end = False
        else:
            self.__end = True

    def _where_empty(self):
        '''return where is empty in the board'''
        return list(zip(*np.where(self.board == 0)))


def _merge(row):
    '''merge the row, there may be some improvement'''
    non_zero = row[row != 0]  # remove zeros
    core = [None]
    for elem in non_zero:
        if core[-1] is None:
            core[-1] = elem
        elif core[-1] == elem:
            core[-1] = 2 * elem
            core.append(None)
        else:
            core.append(elem)
    if core[-1] is None:
        core.pop()
    return core
