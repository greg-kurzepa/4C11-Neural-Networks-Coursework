import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, layer_sizes, activation_function = nn.ReLU, dropout_p = 0, dropout_layers = None, do_batchnorm=False):
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
    
# modified from https://medium.com/@chen-yu/building-a-customized-residual-cnn-with-pytorch-471810e894ed
class ResidualBlock(nn.Module):
    """Keeps the dimension of each input feature the same, but can change the number of channels."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            padding='same', 
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            padding='same', 
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    padding='same', 
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample else x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)

        # print("shape: ", x.shape)

        return x
    
# modified from https://medium.com/@chen-yu/building-a-customized-residual-cnn-with-pytorch-471810e894ed
class ResNet(nn.Module):
    def __init__(self, classes_num: int, in_channels: int = 1, input_size: int = 20):
        super().__init__()

        self.in_channels = in_channels

        # Initial convolution layer
        # input is batch_size * 1 * 20
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=8, 
            kernel_size=3, 
            padding='same', 
            bias=False
        )
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()

        # First block of residual layers and pooling
        # input is batch_size * 8 * 20
        self.layer1 = nn.Sequential(
            ResidualBlock(8, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Second block of residual layers and pooling
        # input is batch_size * 16 * 10
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Flattening and final linear layer
        # input is batch_size * 32 * 5
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=0.5)

        # input is batch_size * 160
        self.linear = nn.Linear(
            in_features=int(160),
            out_features=classes_num
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("SHAPE1: ", x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print("SHAPE2: ", x.shape)

        x = self.layer1(x)
        # print("SHAPE3: ", x.shape)
        x = self.layer2(x)
        # print("SHAPE4: ", x.shape)

        x = self.flatten(x)
        # print("SHAPE5: ", x.shape)
        x = self.dropout(x)
        x = self.linear(x)
        # print("SHAPE6: ", x.shape)

        return x