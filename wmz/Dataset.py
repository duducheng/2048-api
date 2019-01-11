import os
import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

        self.states = []
        self.actions = []

    def __getitem__(self, index):
        return self.states[index], self.actions[index]

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states = []
        self.actions = []

    def load(self, data_dir, file_name=None):
        states_file, actions_file = file_name if file_name is not None \
                                        else 'states.npy', 'actions.npy'

        states_np = np.load(os.path.join(data_dir, states_file))
        self.states.extend([s for s in states_np])

        actions_np = np.load(os.path.join(data_dir, actions_file))
        self.actions.extend(actions_np.tolist())

    def save(self, data_dir, file_name=None):
        states_file, actions_file = file_name if file_name is not None \
                                        else 'states.npy', 'actions.npy'

        np.save(os.path.join(data_dir, states_file), self.states)
        np.save(os.path.join(data_dir, actions_file), self.actions)

    def push(self, new_states, new_actions):
        assert len(new_states) == len(new_actions)

        self.states.extend(new_states)
        self.actions.extend(new_actions)

    def get_loader(self, batch_size, use_gpu):
        kwargs = {'num_workers': 0, 'pin_memory': True} if use_gpu else {}
        return data.DataLoader(self, batch_size=batch_size, shuffle=True, **kwargs)

