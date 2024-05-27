# Handle Library Imports
import os
import sys
import torch
import torch.nn as nn

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Handle Module Imports
from src.models.layers import SpectralPoolNd, ConvBlock, ResConvBlock, GatingSignal, AttentionBlock

# UNET With Skip Connections
class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, features=[16, 32, 64, 128], dropout=0, batchnorm=True, bias=True, pooling='maxpool'):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if pooling.lower() == 'spectralpool':
            self.pool = SpectralPoolNd()
        # Encoder
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(2 * feature, feature, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias))

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        for i in range(0, len(self.decoder), 2):
            # Upsample and concatenate with skip connection
            x = self.decoder[i](x)
            x = torch.cat([x, encoder_outputs[-(i//2 + 1)]], dim=1)
            x = self.decoder[i + 1](x)

        return self.output(x)
    
# Residual UNET With Skip Connections
class ResUNet(nn.Module):

    def __init__(self, in_channels, out_channels, features=[16, 32, 64, 128], dropout=0, batchnorm=True, bias=True, pooling='maxpool'):
        super(ResUNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if pooling.lower() == 'spectralpool':
            self.pool = SpectralPoolNd()
        # Encoder
        for feature in features:
            self.encoder.append(ResConvBlock(in_channels, feature, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ResConvBlock(features[-1], features[-1]*2, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2))
            self.decoder.append(ResConvBlock(2* feature, feature, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias))

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        for i in range(0, len(self.decoder), 2):
            # Upsample and concatenate with skip connection
            x = self.decoder[i](x)
            x = torch.cat([x, encoder_outputs[-(i//2 + 1)]], dim=1)
            x = self.decoder[i + 1](x)

        return self.output(x)
    
# Attention UNET With Skip Connections
class AttentionUNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, features=[16, 32, 64, 128], dropout=0, batchnorm=True, bias=True, pooling='maxpool'):
        super(AttentionUNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if pooling.lower() == 'spectralpool':
            self.pool = SpectralPoolNd()
        # Encoder
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2))
            self.decoder.append(GatingSignal(2 * feature, feature))
            self.decoder.append(AttentionBlock(feature, feature, feature // 2))
            self.decoder.append(ConvBlock(2 * feature, feature, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias))

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        for i in range(0, len(self.decoder), 4):
            # Obtain Gating signal
            gating_sig = self.decoder[i + 1](x)
            # Obtain the Attention Map
            att_x = self.decoder[i + 2](encoder_outputs[-(i//4 + 1)], gating_sig)
            # Upsample the previous output
            x = self.decoder[i](x)
            # Concatenate Attention Map and Conv Output
            x = torch.cat([x, att_x], dim=1)
            x = self.decoder[i + 3](x)

        return self.output(x)
    
# Residual Attention UNET With Skip Connections
class ResAttentionUNet(nn.Module):

    def __init__(self, in_channels, out_channels, features=[16, 32, 64, 128], dropout=0, batchnorm=True, bias=True, pooling='maxpool'):
        super(ResAttentionUNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if pooling.lower() == 'spectralpool':
            self.pool = SpectralPoolNd()
        # Encoder
        for feature in features:
            self.encoder.append(ResConvBlock(in_channels, feature, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ResConvBlock(features[-1], features[-1]*2, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2))
            self.decoder.append(GatingSignal(2 * feature, feature))
            self.decoder.append(AttentionBlock(feature, feature, feature // 2))
            self.decoder.append(ResConvBlock(2 * feature, feature, kernel_size=3, dropout=dropout, batchnorm=batchnorm, bias=bias))

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        for i in range(0, len(self.decoder), 4):
            # Obtain Gating signal
            gating_sig = self.decoder[i + 1](x)
            # Obtain the Attention Map
            att_x = self.decoder[i + 2](encoder_outputs[-(i//4 + 1)], gating_sig)
            # Upsample the previous output
            x = self.decoder[i](x)
            # Concatenate Attention Map and Conv Output
            x = torch.cat([x, att_x], dim=1)
            x = self.decoder[i + 3](x)

        return self.output(x)
