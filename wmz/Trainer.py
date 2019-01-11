import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game2048.game import Game
from wmz.Dataset import Dataset
from wmz.SimpleNet import MODEL_CLASSES
from wmz.utils import MeanRecorder
from wmz.AgentD4 import AgentWrap as TestAgent

GEN_DATA = False
GOAL_SCORES = [128, 256, 512, 1024]
TRAIN_DATA_DIRS = [F"./data/{s}" for s in GOAL_SCORES]

LOAD_MODEL = True
MODEL_TYPE = "V1"
MODEL_PATH = "./model/" + MODEL_TYPE + "/model.pth"
EPOCHS = 5
BATCH_SIZE = 4096 * 8
N_EVAL = 50
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.0005

USE_GPU = torch.cuda.is_available()


class Trainer():
    def __init__(self):
        self.high_score = 600.0
        self.agent_class = MODEL_CLASSES[MODEL_TYPE]
        # data
        self.dataset = Dataset()
        for d in TRAIN_DATA_DIRS:
            self.dataset.load(d)
            print(len(self.dataset))

        # model
        self.model = MODEL_CLASSES[MODEL_TYPE]()
        if USE_GPU:
            self.model.cuda()
        if LOAD_MODEL:
            self.model.load(MODEL_PATH, use_gpu=USE_GPU)

        # train components
        self.criterion = nn.CrossEntropyLoss()
        if USE_GPU:
            self.criterion.cuda()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY)

        self.loss_recorder = MeanRecorder()

    def train(self):
        self.model.train()
        for epoch in range(EPOCHS):
            since = time.time()
            self.loss_recorder.clear('loss')
            for s, a in self.dataset.get_loader(BATCH_SIZE, USE_GPU):
                s = s.float().view(-1, 12, 4, 4)
                a = torch.LongTensor(a).view(-1)
                if USE_GPU:
                    s = s.cuda()
                    a = a.cuda()
                o = self.model(s)

                loss = self.criterion(o, a)
                self.loss_recorder.record('loss', loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(F"Epoch: {epoch}/{EPOCHS}, "
                  F"time(s) = {time.time() - since:.1f}, "
                  F"loss = {self.loss_recorder.mean('loss'):.4f}, ")

        self.model.save(MODEL_PATH)

    def evaluate_accuracy(self):
        self.model.eval()

        acc = cnt = 0
        for s, a in self.dataset.get_loader(BATCH_SIZE, USE_GPU):
            s = s.float().view(-1, 12, 4, 4)
            a = torch.LongTensor(a).view(-1)
            if USE_GPU:
                s = s.cuda()
                a = a.cuda()
            o = self.model(s)

            p = F.softmax(o, dim=1)
            d = torch.argmax(p, dim=1)
            acc += torch.sum(d == a).item()
            cnt += d.shape[0]
        print(F"{acc}/{cnt}={acc / cnt}")

    def evaluate_score(self):
        self.model.eval()

        scores = []
        n_iter = 0
        for i in range(N_EVAL):
            game = Game(4, 2048)
            path = MODEL_PATH if LOAD_MODEL else None
            n_iter += TestAgent(game, model_cls=MODEL_CLASSES[MODEL_TYPE], model_path=path).play()
            scores.append(game.score)

        average_iter = n_iter / N_EVAL
        average_score = sum(scores) / len(scores)
        print(F"Average_iter={average_iter},"
              F"average_score={average_score}",
              Counter(scores))

        if average_score > self.high_score:
            self.high_score = average_score
            self.model.save(F"./model/model_{average_score}.pth")
