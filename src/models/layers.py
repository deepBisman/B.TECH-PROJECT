# Handle Library Imports
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import sigpy as sp

# Dynamically add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Handle Module imports
from src.utils.signal_utils import crop_forward, pad_backward

# Spectral Pooling Layer
class SpectralPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, return_shape):
        input = sp.from_pytorch(input)
        ctx.input_shape = input.shape
        output = crop_forward(input, return_shape)
        output = sp.to_pytorch(output)
        output = output.float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = sp.from_pytorch(grad_output)
        grad_input = pad_backward(grad_output, ctx.input_shape)
        grad_input = sp.to_pytorch(grad_input)
        grad_input = grad_input.float()
        return grad_input, None

class SpectralPoolNd(nn.Module):
    def forward(self, input):
        return_shape = [input.shape[-2] // 2, input.shape[-1] // 2]
        return SpectralPoolingFunction.apply(input, return_shape)
    
# Convolutional Block 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0, batchnorm=True, bias = True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", bias=bias)
        self.batchnorm1 = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", bias=bias)
        self.batchnorm2 = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        if self.batchnorm1:
            x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        if self.batchnorm2:
            x = self.batchnorm2(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    
# Residual convolutional block
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0, batchnorm=True, bias=True):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", bias=bias)
        self.batchnorm1 = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding="same", bias=bias)
        self.batchnorm2 = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same", bias=bias)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        if self.batchnorm1:
            x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        if self.batchnorm2:
            x = self.batchnorm2(x)
        if self.dropout:
            x = self.dropout(x)
        x += shortcut
        return x

# Gating Signal
class GatingSignal(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True, bias=True):
        super(GatingSignal, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else None

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        if self.batchnorm:
            x = self.batchnorm(x)
        return x
    
# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.theta_x = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0, bias = False)
        self.phi_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x, gating):
        # theta => (b, c, h, w) -> (b, i_c, h, w) -> (b, i_c, h, w)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta_x(x)
        theta_x_size = theta_x.size()

        # g (b, c, h', w') -> phi_g (b, i_c, h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, h, w) -> (b, i_c, h/s1, w/s2)
        phi_g = F.interpolate(self.phi_g(gating), size=theta_x_size[2:] ,mode='bilinear')
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c h/s1, w/s2)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=x.size()[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x) * x
        attenblock = self.W(y)
        return attenblock
