# taken from https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3/

import torch
import torch.nn as nn

relu = nn.ReLU()

class UNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.example_input_array = torch.zeros(1, n_class, 32, 32)

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: -1 x 1 x 32 x 32
        ch1 = 32
        self.e11 = nn.Conv2d(n_class, ch1, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(ch1, ch1, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: -1 x 32 x 16 x 16
        ch2 = 64
        self.e21 = nn.Conv2d(ch1, ch2, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(ch2, ch2, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: -1 x 64 x 8 x 8
        ch3 = 128
        self.e31 = nn.Conv2d(ch2, ch3, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(ch3, ch3, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: -1 x 128 x 4 x 4
        ch4 = 256
        self.e41 = nn.Conv2d(ch3, ch4, kernel_size=3, padding=1) # output: 30x30x1024
        self.e42 = nn.Conv2d(ch4, ch4, kernel_size=3, padding=1) # output: 28x28x1024

        # Decoder
        # input: -1 x 256 x 4 x 4
        self.upconv1 = nn.ConvTranspose2d(ch4, ch3, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(ch4, ch3, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(ch3, ch3, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(ch3, ch2, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(ch3, ch2, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(ch2, ch2, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(ch2, ch1, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(ch2, ch1, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(ch1, ch1, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(ch1, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))

        # Decoder
        xu1 = self.upconv1(xe42)
        xu11 = torch.cat([xu1, xe32], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe22], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe12], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        # Output layer
        out = self.outconv(xd32)

        return out