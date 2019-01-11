import numpy as np
import torch

from game2048.agents import Agent
from wmz.SimpleNet import MODEL_CLASSES
from wmz.utils import onehot

USE_GPU = True
DEVICE_ID = 0
torch.cuda.set_device(DEVICE_ID)


class AgentWrap(Agent):
    def __init__(self, game, display=None, model=None, model_cls=None, model_path=None):
        super(AgentWrap, self).__init__(game, display)

        if model is None:
            self.model = model_cls()
        else:
            self.model = model
        if USE_GPU:
            self.model.cuda()
        if model_path is not None:
            self.model.load(model_path, USE_GPU)

    def step(self):
        self.model.eval()
        onehot_board = onehot(self.game.board)
        ds = np.zeros(shape=(4,))
        for i in range(4):
            state = np.rot90(onehot_board, k=i, axes=(1, 2)).copy()
            d = self.model.predict(state, use_gpu=USE_GPU)
            ds[(d - i) % 4] += 1
        return int(np.argmax(ds))


class AgentD4(Agent):
    def __init__(self, game, display=None):
        super(AgentD4, self).__init__(game, display)

        self.models = dict()
        # self.add_model("V1", "V1_1", "./model/V1/model_1013.76.pth")

        # self.add_model("V2", "V2_1", "./model/V2/model_1070.08.pth")
        # self.add_model("V2", "V2_2", "./model/V2/model_1100.8.pth")
        self.add_model("V2", "V2_3", "./model/V2/model_1086.08.pth")
        self.add_model("V2", "V2_4", "./model/V2/model_1111.04.pth")

        self.add_model("V3", "V3_1", "./model/V3/model.pth")
        # self.add_model("V3", "V3_2", "./model/V3/model_1036.8.pth")
        # self.add_model("V3", "V3_3", "./model/V3/model_1080.32.pth")
        # self.add_model("V3", "V3_4", "./model/V3/model_1105.92.pth")
        self.add_model("V3", "V3_5", "./model/V3/model_1187.84.pth")
        self.add_model("V3", "V3_6", "./model/V3/model_1308.16.pth")

        self.weights = {
            'V1_1': 3.349426807760141,
            'V2_1': 3.426477221470456,
            'V2_2': 3.397931227285435,
            'V2_3': 3.4536413491319293,
            'V2_4': 3.457669549385526,
            'V3_1': 3.4995825055425986,
            'V3_2': 3.3894977550963494,
            'V3_3': 3.4214615471230716,
            'V3_4': 3.492744044369091,
            'V3_5': 3.656888407497668,
            'V3_6': 3.717055567448546,
        }

    def add_model(self, model_type, key, model_path):
        self.models[key] = MODEL_CLASSES[model_type]()
        self.models[key].eval()
        if USE_GPU:
            self.models[key].cuda()
        if model_path is not None:
            self.models[key].load(model_path, USE_GPU)

    def vote(self, onehot_board, keys, weights=None):
        if weights is None:
            weights = {key: 1.0 for key in keys}

        # onehot_board_flip = np.flip(onehot_board, axis=1).copy()
        ds = np.zeros(shape=(4,))
        for key in keys:
            for i in range(4):
                state = np.rot90(onehot_board, k=i, axes=(1, 2)).copy()
                d = self.models[key].predict(state, use_gpu=USE_GPU)
                ds[(d - i) % 4] += weights[key]

            # for i in range(4):
            #     state = np.rot90(onehot_board_flip, k=i, axes=(1, 2)).copy()
            #     d = self.models[key].predict(state, use_gpu=USE_GPU)
            #     ds[[0, 3, 2, 1][(d - i) % 4]] += 1

        return ds

    def step(self):
        onehot_board = onehot(self.game.board)

        ds = self.vote(onehot_board, keys=self.models.keys(), weights=self.weights)
        return int(np.argmax(ds))

    def count_error(self):
        from game2048.game import Game
        from game2048.expectimax import board_to_move

        err_counts = {key: [0, 0, 0] for key in self.models.keys()}
        err_counts["tot"] = [0, 0, 0]

        scores = []
        for _ in range(50):
            game = Game(4, 2048)
            while not game.end:
                if self.display is not None:
                    self.display.display(game)

                s, a = onehot(game.board), board_to_move(game.board)

                ds = np.zeros(shape=(4,))
                for key in self.models.keys():
                    for i in range(4):
                        state = np.rot90(s, k=i, axes=(1, 2)).copy()
                        d = self.models[key].predict(state, use_gpu=USE_GPU)
                        ds[(d - i) % 4] += self.weights[key]

                        if (d - i) % 4 != a:
                            err_counts[key][0] += 1
                    err_counts[key][1] += 4

                err_counts["tot"][1] += 1
                if int(np.argmax(ds)) != a:
                    err_counts["tot"][0] += 1

                game.move(int(np.argmax(ds)))
            print(F"{_}/{50}:{game.score}")
            scores.append(game.score)

        for k, e in err_counts.items():
            e[2] = e[0] / e[1]

        return scores, err_counts

        # Counter({1024: 28, 2048: 14, 512: 7, 256: 1})
        # {'V1_1': [36288, 121544, 0.2985585466991378],
        #  'V2_1': [35472, 121544, 0.2918449285855328],
        #  'V2_2': [35770, 121544, 0.29429671559270715],
        #  'V2_3': [35193, 121544, 0.2895494635687488],
        #  'V2_4': [35152, 121544, 0.2892121371684328],
        #  'V3_1': [34731, 121544, 0.2857483709603107],
        #  'V3_2': [35859, 121544, 0.2950289607055881],
        #  'V3_3': [35524, 121544, 0.2922727571908116],
        #  'V3_4': [34799, 121544, 0.2863078391364444],
        #  'V3_5': [33237, 121544, 0.27345652603172516],
        #  'V3_6': [32699, 121544, 0.26903014546172577],
        #  'tot': [7461, 30386, 0.2455407095372869]}
        # {
        # 'V1_1': 3.349426807760141,
        #  'V2_1': 3.426477221470456,
        #  'V2_2': 3.397931227285435,
        #  'V2_3': 3.4536413491319293,
        #  'V2_4': 3.457669549385526,
        #  'V3_1': 3.4995825055425986,
        #  'V3_2': 3.3894977550963494,
        #  'V3_3': 3.4214615471230716,
        #  'V3_4': 3.492744044369091,
        #  'V3_5': 3.656888407497668,
        #  'V3_6': 3.717055567448546,
        #  'tot': 4.0726444176383865
        #  }
        #
        # Counter({1024: 26, 2048: 17, 512: 4, 256: 3})
        # {'V1_1': [36563, 123868, 0.29517712403526336],
        # 'V2_1': [35755, 123868, 0.2886540510866406],
        # 'V2_2': [36158, 123868, 0.29190751445086704],
        # 'V2_3': [35571, 123868, 0.28716859883101364],
        # 'V2_4': [35312, 123868, 0.285077663319017],
        # 'V3_1': [35242, 123868, 0.284512545613072],
        # 'V3_2': [36085, 123868, 0.2913181774146672],
        # 'V3_3': [35964, 123868, 0.2903413310943908],
        # 'V3_4': [35352, 123868, 0.28540058772241417],
        # 'V3_5': [33792, 123868, 0.27280653598992477],
        # 'V3_6': [32985, 123868, 0.26629153615138695],
        # 'tot': [7429, 30967, 0.23990053928375368]}
        # {
        # 'V1_1': 3.387796406202992,
        # 'V2_1': 3.4643546357152846,
        # 'V2_2': 3.4257425742574257,
        # 'V2_3': 3.4822748868460263,
        # 'V2_4': 3.507816039873131,
        # 'V3_1': 3.51478349696385,
        # 'V3_2': 3.4326728557572403,
        # 'V3_3': 3.444221999777555,
        # 'V3_4': 3.503847024213623,
        # 'V3_5': 3.6656013257575757,
        # 'V3_6': 3.755282704259512,
        # 'tot': 4.168394131107821}
