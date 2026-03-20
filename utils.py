import sys
import pickle
import torch


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


class presDataset(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.pS_array, self.pH_array = a, b
    def __getitem__(self, idx):
        sid = self.pS_array[idx]
        hid = self.pH_array[idx]
        return sid, hid

    def __len__(self):
        return self.pH_array.shape[0]