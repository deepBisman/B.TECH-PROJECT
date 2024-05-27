# models/composite.py
# Handle Library Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Two Stage Regression : Model Type 1 - TSR-M1
class TwoStageUNetSequential(nn.Module):
    def __init__(self, segmentation_net, regression_net):
        super(TwoStageUNetSequential, self).__init__()
        self.segmentation_net = segmentation_net # Returns logits
        self.regression_net = regression_net

        # Freeze segmentation_net
        for param in self.segmentation_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1 = torch.round(F.sigmoid(self.segmentation_net(x)))
        x2 = self.regression_net(torch.cat([x, x1], dim=1))
        return x2

# Two Stage Regression : Model Type 2 - TSR-M2
class TwoStageUNetFusion(nn.Module):
    def __init__(self, segmentation_net, regression_net):
        super(TwoStageUNetFusion, self).__init__()
        self.segmentation_net = segmentation_net # Returns logitss
        self.regression_net = regression_net

        # Freeze segmentation_net
        for param in self.segmentation_net.parameters():
            param.requires_grad = False


    def forward(self, x):
        x1 = torch.round(F.sigmoid(self.segmentation_net(x)))
        x2 = self.regression_net(x)
        # Fuse the outputs by element wise multiplication
        x_out = x1 * x2
        return x_out
