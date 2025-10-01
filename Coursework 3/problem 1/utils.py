import torch
import torch.nn as nn
import numpy as np
import h5py
import scipy.io

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# # Normaliser
class MinMaxNormaliser():
    def __init__(self, data):
        self.max = data.max()
        self.min = data.min()

    def normalise(self, x):
        return (x - self.min) / (self.max - self.min)
    
    def denormalise(self, x):
        return x * (self.max - self.min) + self.min

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_input, layer_hidden):
        super(RNN, self).__init__()

        self.layers = nn.ModuleList()
        for j in range(len(layer_input) - 1):
            self.layers.append(nn.Linear(layer_input[j], layer_input[j + 1]))
            if j != len(layer_input) - 1:
                self.layers.append(nn.SELU())

        self.hidden_layers = nn.ModuleList()
        self.hidden_size   = hidden_size

        for j in range(len(layer_hidden) - 1):
            self.hidden_layers.append(nn.Linear(layer_hidden[j], layer_hidden[j + 1]))
            if j != len(layer_hidden) - 1:
                self.hidden_layers.append(nn.SELU())

    def forward(self, input, output, hidden, dt):
        h0 = hidden
        h = torch.cat((output, hidden), 1)
        for _, m in enumerate(self.hidden_layers):
            h = m(h)

        h = h*dt + h0
        combined = torch.cat((output, (output-input)/dt, hidden), 1)
        x = combined
        for _, l in enumerate(self.layers):
            x = l(x)

        output = x.squeeze(1)
        hidden = h
        return output, hidden

    def initHidden(self,b_size):

        return torch.zeros(b_size, self.hidden_size)