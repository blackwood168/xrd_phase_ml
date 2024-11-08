import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=1, features_g=64):
        super().__init__()
        
        # Encoder (Downsampling)
        self.encoder = nn.Sequential(
            self._make_encoder_block(in_channels, features_g),
            self._make_encoder_block(features_g, features_g * 2),
            self._make_encoder_block(features_g * 2, features_g * 4),
            self._make_encoder_block(features_g * 4, features_g * 4)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(features_g * 4, features_g * 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(features_g * 4),
            nn.GELU(),
            nn.Conv3d(features_g * 4, features_g * 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(features_g * 4),
            nn.GELU(),
        )
        
        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            self._make_decoder_block(features_g * 4, features_g * 2),
            self._make_decoder_block(features_g * 2, features_g),
            self._make_decoder_block(features_g, features_g),
            nn.Conv3d(features_g, in_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool3d((26, 18, 23))
        )

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        # Get input mask (1 where input is non-zero)
        mask = (x != 0).float()
        
        # Encode-decode
        features = self.encoder(x)
        features = self.bottleneck(features)
        output = self.decoder(features)
        
        # Preserve input values where mask is 1
        return output * (1 - mask) + x * mask

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features_d=32):
        super().__init__()
        
        self.main = nn.Sequential(
            # Focus on edge regions
            self._make_disc_block(in_channels, features_d, batch_norm=False),
            self._make_disc_block(features_d, features_d * 2),
            self._make_disc_block(features_d * 2, features_d * 4),
            self._make_disc_block(features_d * 4, features_d * 8),
            nn.Conv3d(features_d * 8, 1, kernel_size=3, stride=1, padding=1),
        )
        
        #self.activation = nn.Sigmoid()

    def _make_disc_block(self, in_channels, out_channels, batch_norm=True):
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=1, padding=1)]
        if batch_norm:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.main(x)
        # Global average pooling and activation
        return torch.mean(features, dim=[2,3,4])