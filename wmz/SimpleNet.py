import torch
import torch.nn as nn
import torch.nn.functional as F

from wmz.utils import StorableNet


class SimpleNet(StorableNet):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv = None
        self.fc = None

    def forward(self, x):
        o = [c(x) for c in self.conv]
        o = torch.cat([i.view(i.shape[0], -1) for i in o], 1)
        return self.fc(o)

    def predict(self, state, use_gpu=False):
        state = torch.Tensor(state).view(1, 12, 4, 4)
        if use_gpu:
            state = state.cuda()
        p = F.softmax(self.forward(state), dim=1)
        return int(torch.argmax(p))


class SimpleNetV1(SimpleNet):
    def __init__(self):
        super(SimpleNetV1, self).__init__()

        num_feature = 64
        self.conv = nn.ModuleList()
        for ks in [(1, 4), (4, 1), (2, 2), (3, 3), (4, 4)]:
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(12, num_feature, kernel_size=ks),
                    nn.BatchNorm2d(num_feature),
                    nn.ReLU(),
                ))

        self.fc = nn.Sequential(
            nn.Linear(num_feature * (4 + 4 + 9 + 4 + 1), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )


class SimpleNetV2(SimpleNet):
    def __init__(self):
        super(SimpleNetV2, self).__init__()

        num_feature = 128
        self.conv = nn.ModuleList()
        for ks in [(1, 4), (4, 1), (2, 2), (3, 3), (4, 4)]:
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(12, num_feature, kernel_size=ks),
                    nn.BatchNorm2d(num_feature),
                    nn.ReLU(),
                ))

        self.fc = nn.Sequential(
            nn.Linear(num_feature * (4 + 4 + 9 + 4 + 1), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )


class SimpleNetV3(SimpleNet):
    def __init__(self):
        super(SimpleNetV3, self).__init__()

        num_feature = 128
        self.conv = nn.ModuleList()
        for ks in [(1, 4), (4, 1), (2, 2), (3, 3), (4, 4)]:
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(12, num_feature, kernel_size=ks),
                    nn.BatchNorm2d(num_feature),
                    nn.ReLU(),
                ))

        self.fc = nn.Sequential(
            nn.Linear(num_feature * (4 + 4 + 9 + 4 + 1), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )


MODEL_CLASSES = {
    "V1": SimpleNetV1,
    "V2": SimpleNetV2,
    "V3": SimpleNetV3,
}
