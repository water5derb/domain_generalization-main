import torch
import torch.nn as nn
from models.modules import ConvBlock, FullyConnectedBlock
from gradient_reversal import GradientReversal
from gradient_reversal import ReverseLayerF


class CNNModel(nn.Module):
    def __init__(self, class_label=10, domain_label=5):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential(
            ConvBlock(input_channel=1, output_channel=32),
            ConvBlock(input_channel=32, output_channel=64),
            ConvBlock(input_channel=64, output_channel=128),
            ConvBlock(input_channel=128, output_channel=128),
            nn.Flatten()
        )

        self.class_classifier = nn.Sequential(
            FullyConnectedBlock(in_features=11520, out_features=512),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=512, out_features=128),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=128, out_features=64),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=64, out_features=class_label)
        )

        self.domain_classifier = nn.Sequential(
            FullyConnectedBlock(in_features=11520, out_features=512),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=512, out_features=64),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=64, out_features=domain_label)
        )

    def forward(self, x, alpha):
        feature = self.feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_y = self.class_classifier(feature)
        domain_y = self.domain_classifier(reverse_feature)
        return class_y, domain_y


class SounAdapterCNN(nn.Module):
    def __init__(self, class_label=10):
        super(SounAdapterCNN, self).__init__()
        self.feature = nn.Sequential(
            ConvBlock(input_channel=1, output_channel=16),
            ConvBlock(input_channel=16, output_channel=32),
            ConvBlock(input_channel=32, output_channel=64),
            nn.Flatten()
        )

        self.class_classifier = nn.Sequential(
            FullyConnectedBlock(in_features=59520, out_features=100),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=100, out_features=50),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=50, out_features=class_label)
        )
    
    def forward(self, x):
        x = self.feature(x)
        x = self.class_classifier(x)
        return x


class CNNDeep(nn.Module):
    # input (1, 513, 1034) -> flatten 512*30*3 = 46080
    def __init__(self, class_label=10):
        super(CNNDeep, self).__init__()
        self.feature = nn.Sequential(
            ConvBlock(input_channel=1, output_channel=64),
            ConvBlock(input_channel=64, output_channel=128),
            ConvBlock(input_channel=128, output_channel=256),
            ConvBlock(input_channel=256, output_channel=512),
            nn.Flatten()
        )

        self.class_classifier = nn.Sequential(
            FullyConnectedBlock(in_features=46080, out_features=512),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=512, out_features=128),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=128, out_features=class_label)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.class_classifier(x)
        return x


class FeatureMLP(nn.Module):
    # No CNN — pools raw spectrogram into mean+std stats over time, then MLP
    # input (1, 513, 1034) -> mean & std over time -> (1026,) -> MLP
    def __init__(self, class_label=10):
        super(FeatureMLP, self).__init__()
        self.classifier = nn.Sequential(
            FullyConnectedBlock(in_features=1026, out_features=512),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=512, out_features=128),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=128, out_features=class_label)
        )

    def forward(self, x):
        # x: (batch, 1, 513, 1034)
        mean = x.mean(dim=-1)   # (batch, 1, 513)
        std  = x.std(dim=-1)    # (batch, 1, 513)
        feat = torch.cat([mean, std], dim=-1).squeeze(1)  # (batch, 1026)
        return self.classifier(feat)


class HybridCNNFeature(nn.Module):
    # CNN branch (shallow) + statistical feature branch, concatenated before classifier
    # input (1, 513, 1034)
    def __init__(self, class_label=10):
        super(HybridCNNFeature, self).__init__()
        self.cnn_branch = nn.Sequential(
            ConvBlock(input_channel=1, output_channel=32),
            ConvBlock(input_channel=32, output_channel=64),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )  # -> 64*8*8 = 4096

        self.stat_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1026, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )  # mean+std over time -> 256

        self.classifier = nn.Sequential(
            FullyConnectedBlock(in_features=4096 + 256, out_features=512),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=512, out_features=128),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=128, out_features=class_label)
        )

    def forward(self, x):
        cnn_feat = self.cnn_branch(x)                              # (batch, 4096)
        mean = x.mean(dim=-1)                                      # (batch, 1, 513)
        std  = x.std(dim=-1)                                       # (batch, 1, 513)
        stat_feat = torch.cat([mean, std], dim=-1).squeeze(1)      # (batch, 1026)
        stat_feat = self.stat_branch(stat_feat)                    # (batch, 256)
        combined = torch.cat([cnn_feat, stat_feat], dim=-1)        # (batch, 4352)
        return self.classifier(combined)


