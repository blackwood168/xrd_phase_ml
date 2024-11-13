from torch import nn
import torch


class DoubleConv(nn.Module):
    """Double convolution block with instance normalization and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double convolution block."""
        return self.double_conv(x)


class ResFBlock(nn.Module):
    """Residual block with FFT processing."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the FFT residual block.
        
        Args:
            in_channels: Number of input channels (must equal out_channels)
            out_channels: Number of output channels
        """
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.real_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FFT residual block."""
        x_skip = x
        x_conv = self.double_conv(x)
        x_real = torch.abs(torch.fft.rfftn(x, s=x.shape[2:], dim=(-3, -2, -1)))
        x_real = self.real_conv(x_real)
        return x_skip + x_conv + torch.fft.irfftn(x_real, s=x.shape[2:], dim=(-3, -2, -1))


class DownBlock(nn.Module):
    """Downsampling block with FFT processing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.fft_block = ResFBlock(out_channels, out_channels)
        self.down_sample = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the downsampling block.
        
        Returns:
            Tuple of (downsampled output, skip connection output)
        """
        skip_out = self.fft_block(self.double_conv(x))
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    """Upsampling block with FFT processing."""

    def __init__(self, in_channels: int, out_channels: int, up_sample_mode: str) -> None:
        """Initialize the upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            up_sample_mode: Upsampling mode (currently only supports 'trilinear')
        """
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.fft_block = ResFBlock(out_channels, out_channels)

    def forward(self, down_input: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling block.
        
        Args:
            down_input: Input from the encoder path
            skip_input: Skip connection input
            
        Returns:
            Processed and upsampled output
        """
        x = nn.Upsample(size=skip_input.shape[2:], mode='trilinear', align_corners=False)(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.fft_block(self.double_conv(x))


class UNet_FFT(nn.Module):
    """U-Net architecture enhanced with FFT processing."""

    def __init__(self, num_layers: int = 2, out_classes: int = 1, up_sample_mode: str = 'trilinear') -> None:
        """Initialize the FFT-enhanced U-Net model.
        
        Args:
            num_layers: Number of down/up sampling layers
            out_classes: Number of output classes/channels
            up_sample_mode: Upsampling mode for decoder path
        """
        super().__init__()
        self.up_sample_mode = up_sample_mode
        self.num_layers = num_layers

        # Calculate filter sizes for each layer
        initial_filters = 64
        self.filters = [initial_filters * (2**i) for i in range(num_layers)]
        
        # Encoder path
        self.down_blocks = nn.ModuleList()
        in_channels = 1
        for filters in self.filters:
            self.down_blocks.append(DownBlock(in_channels, filters))
            in_channels = filters

        # Bottleneck
        self.double_conv = DoubleConv(self.filters[-1], self.filters[-1] * 2)

        # Decoder path
        self.up_blocks = nn.ModuleList()
        in_channels = self.filters[-1] * 2
        for filters in reversed(self.filters):
            self.up_blocks.append(UpBlock(in_channels + filters, filters, self.up_sample_mode))
            in_channels = filters

        # Final convolution
        self.conv_last = nn.Conv3d(self.filters[0], out_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FFT-enhanced U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through the U-Net
        """
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_connections.append(skip)
            
        # Bottleneck
        x = self.double_conv(x)
        
        # Decoder path
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = up_block(x, skip)
            
        return self.conv_last(x)