import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
            self,
            input_channel,
            output_channel,
            kernel=3,
            stride=1,
            padding=0
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.MaxPool2d(kernel_size=(2, 4)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnectedBlock, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(num_features=out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc_block(x)
        return x