import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, layer_sizes, activation_function = nn.ReLU, dropout_p = 0, dropout_layers = None, do_batchnorm=False):
        """
        Note, input layer is 0, first hidden layer is 1. So dropout_layers=[1] would put a dropout in the first hidden layer.

        Args:
            layer_sizes (_type_): _description_
            activation_function (_type_, optional): _description_. Defaults to nn.ReLU.
            dropout_p (int, optional): _description_. Defaults to 0.
            dropout_layers (_type_, optional): _description_. Defaults to None.
            do_batchnorm (bool, optional): _description_. Defaults to False.
        """
        super(FCNN, self).__init__()

        self.n_transitions = len(layer_sizes) - 1
        assert self.n_transitions >= 1
        self.layers = nn.ModuleList()

        if dropout_p: print(f"Using dropout with probability p={dropout_p}")

        for i in range(self.n_transitions):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i != self.n_transitions - 1:
                if do_batchnorm: self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                self.layers.append(activation_function())

            # if dropout_layers is None, add dropout layer in all but input and output layers
            if dropout_layers is None:
                if i != self.n_transitions - 1 and dropout_p != 0:
                    self.layers.append(nn.Dropout(dropout_p))
            else:
                # if dropout_layers is specified, add dropout layer in specified layers
                if i+1 in dropout_layers and dropout_p != 0:
                    self.layers.append(nn.Dropout(dropout_p))

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
    
class LSTMWithLinear(nn.Module):
    """An LSTM model with n_hidden hidden units.
    LSTM output is the hidden states (there are as many outputs as hidden states).
    Thus there is also a linear layer to map to one output (or the other desired output size).
    
    input_size is NOT the sequence length, it is the size of each sequence element. Any sequence length can be input.
    """

    def __init__(self, hidden_size, output_size, input_size=1, num_layers=1, dropout=0):
        super(LSTMWithLinear, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # input is (batch_size, seq_length, input_size)
        x, _ = self.lstm(x) # output is (batch_size, seq_length, hidden_size)
        # x = x[:, -1, :] # select the hidden state from only the final time step

        # now we put this through a linear layer. both the first 2 dimensions are a batch size;
        # the linear layer operates within the last dimension only.
        x = self.fc(x)

        return x