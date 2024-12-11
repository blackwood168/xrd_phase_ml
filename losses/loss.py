from torch import nn
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class TorchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=True, spatial_dims=3)

    def forward(self, preds, target):
        return self.mse_loss(preds, target) #+ 0.001*(1 - self.ssim(preds, target))
    
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=True, spatial_dims=3)

    def forward(self, preds, target):
        return 1 - self.ssim(preds, target)
