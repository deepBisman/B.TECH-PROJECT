# Handle Library Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

######### Segmentation Loss Functions
# Dice Coefficient Metric
class DiceScore(nn.Module):
    def __init__(self):
        super(DiceScore, self).__init__()

    def forward(self, y_pred, y_true):
        # Flatten the predictions and targets
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        
        # Convert predictions to binary
        y_pred = torch.round(y_pred)
        
        # Calculate intersection and union
        intersection = torch.sum(y_pred * y_true)
        coefficient = (2.0 * intersection + 1) / (torch.sum(y_pred) + torch.sum(y_true) + 1)
        
        return coefficient

# Dice Loss with Logits
class DiceWithLogitsLoss(nn.Module):
    def __init__(self):
        super(DiceWithLogitsLoss, self).__init__()
        self.dice = DiceScore()

    def forward(self, input, target):
        dice_score = self.dice(torch.sigmoid(input), target)
        loss = 1 - dice_score  # Use (1 - dice_score) as the dice loss
        return loss

# Focal Loss with Logits
class FocalWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalWithLogitsLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # Apply the sigmoid function to the logits to get probabilities
        p_t = torch.sigmoid(input)
        eps = 1e-20
        # Calculate Focal Loss
        if self.alpha is not None:
            loss = - ((target * (self.alpha[1] * ((1 - p_t) ** self.gamma) * torch.log(p_t + eps))) + ((1 - target) * (self.alpha[0] * (p_t ** self.gamma) * torch.log(1 - p_t + eps))))
        else:
            loss = - ((target * (((1 - p_t) ** self.gamma) * torch.log(p_t + eps))) + ((1 - target) * ((p_t ** self.gamma) * torch.log(1 - p_t + eps))))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Dice + BCE Loss with Logits
class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceScore()

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        dice_score = self.dice(torch.sigmoid(input), target)
        loss = bce_loss + (1 - dice_score)  # Use (1 - dice_score) as the dice loss
        return loss

# Dice + Focal Loss with Logits
class DiceFocalWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(DiceFocalWithLogitsLoss, self).__init__()
        self.focal = FocalWithLogitsLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceScore()

    def forward(self, input, target):
        focal_loss = self.focal(input, target)
        dice_score = self.dice(torch.sigmoid(input), target)
        loss = focal_loss + (1 - dice_score)  # Use (1 - dice_score) as the dice loss
        return loss

# Dice + Focal + BCE Loss with Logits
class DiceFocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(DiceFocalBCEWithLogitsLoss, self).__init__()
        self.focal = FocalWithLogitsLoss(gamma=gamma, alpha=alpha)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceScore()

    def forward(self, input, target):
        focal_loss = self.focal(input, target)
        bce_loss = self.bce(input, target)
        dice_score = self.dice(torch.sigmoid(input), target)
        loss = bce_loss + focal_loss + (1 - dice_score)  # Use (1 - dice_score) as the dice loss
        return loss

######### Regression Loss Functions
# Image Gradient Loss
class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        # Sobel kernel for the gradient map calculation
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, output, target):
        # Calculate gradient maps of x and y directions for target and output
        gx_target = F.conv2d(target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(output, self.kernel_y, stride=1, padding=1)
        
        # Compute gradient loss as MSE between gradient maps
        grad_loss = F.mse_loss(gx_target, gx_output) + F.mse_loss(gy_target, gy_output)
        return grad_loss

# SSIM Loss
class SSIMLoss(nn.Module):
    def __init__(self):
        super(GradSSIMLoss, self).__init__()

    def forward(self, img1, img2):
        # Calculate the SSIM loss between two images
        loss = 1 - ssim(img1, img2)
        return loss

# Grad + SSIM Loss
class GradSSIMLoss(nn.Module):
    def __init__(self):
        super(GradSSIMLoss, self).__init__()
        self.grad_loss = GradLoss()

    def forward(self, img1, img2):
        # Calculate the SSIM loss between two images
        ssim_loss = 1 - ssim(img1, img2)
        grad_loss = self.grad_loss(img1, img2)
        loss = ssim_loss + grad_loss
        return loss

# MSE + Grad Loss
class MSEGradLoss(nn.Module):
    def __init__(self):
        super(MSEGradLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.grad_loss = GradLoss()

    def forward(self, output, target):
        mse_loss = self.mse_loss(output, target)
        grad_loss = self.grad_loss(output, target)
        loss = mse_loss + grad_loss
        return loss

# MSE + SSIM Loss
class MSESSIMLoss(nn.Module):
    def __init__(self):
        super(MSESSIMLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, img1, img2):
        mse_loss = self.mse_loss(img1, img2)
        ssim_loss = 1 - ssim(img1, img2)
        loss = mse_loss + ssim_loss
        return loss

# MSE + SSIM + Grad Loss
class MSESSIMGradLoss(nn.Module):
    def __init__(self):
        super(MSESSIMGradLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.grad_loss = GradLoss()

    def forward(self, img1, img2):
        mse_loss = self.mse_loss(img1, img2)
        ssim_loss = 1 - ssim(img1, img2)
        grad_loss = self.grad_loss(img1, img2)
        loss = mse_loss + ssim_loss + grad_loss
        return loss
