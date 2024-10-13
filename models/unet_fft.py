from torch import nn
import torch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)
class ResFBlock(nn.Module):
    def __init__(self, in_channels, out_channels): #in and out must be the same...
        super(ResFBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.real_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )
    def forward(self, x):
        x_skip = x
        x_conv = self.double_conv(x)
        x_real = torch.abs(torch.fft.rfftn(x, s = x.shape[2:], dim = (-3, -2, -1)))
        x_real = self.real_conv(x_real)
        return x_skip + x_conv + torch.fft.irfftn(x_real, s = x.shape[2:], dim  = (-3, -2, -1))
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.fft_block = ResFBlock(out_channels, out_channels)
        self.down_sample = nn.MaxPool3d(2)

    def forward(self, x):
        skip_out = self.fft_block(self.double_conv(x))
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        #if up_sample_mode == 'conv_transpose':
        #    self.up_sample = nn.ConvTranspose3d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
        #elif up_sample_mode == 'trilinear':
        #    self.up_sample = nn.Upsample(size = (8, 9, 7), mode='trilinear', align_corners=False)
        #else:
        #    raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `trilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.fft_block = ResFBlock(out_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = nn.Upsample(size = skip_input.shape[2:], mode = 'trilinear', align_corners=False)(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.fft_block(self.double_conv(x))

class UNet_FFT(nn.Module):
    def __init__(self, out_classes = 1, up_sample_mode='trilinear'):
        super().__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64)
        self.down_conv2 = DownBlock(64, 128)
        #self.down_conv3 = DownBlock(128, 256)
        #self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(128, 256)
        # Upsampling Path
        #self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        #self.up_conv3 = UpBlock(256 + 128, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(256 + 128, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv3d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        #x, skip3_out = self.down_conv3(x)
        #x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        #x = self.up_conv4(x, skip4_out)
        #x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x
