from torch import nn
import torch


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the double convolution block."""
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downsampling block with double convolution and max pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the downsampling block.
        
        Returns:
            Tuple of (downsampled output, skip connection output)
        """
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    """Upsampling block with double convolution and skip connections."""

    def __init__(self, in_channels: int, out_channels: int, up_sample_mode: str) -> None:
        """Initialize the upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            up_sample_mode: Upsampling mode ('trilinear' only currently supported)
        """
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the upsampling block.
        
        Args:
            down_input: Input from the downsampling path
            skip_input: Skip connection input from the corresponding downsampling block
            
        Returns:
            Upsampled and convolved output
        """
        x = nn.Upsample(
            size=skip_input.shape[2:],
            mode='trilinear',
            align_corners=False
        )(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f'pixel_shuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)'
            )
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(
                f'pixel_shuffle expects its input\'s \'channel\' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}'
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(
            *x.shape[:-4],
            nOut,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)
import math
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 8 * num_feat, 3, 1, 1))
                m.append(PixelShuffle3d(2))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(PixelShuffle3d(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
class SuperResolutionUnet(nn.Module):
    """Mini U-Net architecture for 3D image processing."""

    def __init__(self, num_layers: int = 2, out_classes: int = 1, up_sample_mode: str = 'trilinear') -> None:
        """Initialize the Mini U-Net model.
        
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
        self.conv_before_upsample = nn.Conv3d(self.filters[0], out_classes, kernel_size=3, padding=1) #added this
        # Final convolution
        self.conv_last = nn.Conv3d(out_classes, out_classes, kernel_size=1)

        self.final_upsample = Upsample( #new upsample
            scale=2,
            num_feat=out_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Mini U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through the U-Net
        """
        # Store skip connections
        skip_connections = []
        if x.max() > 1:
            x = x / x.max()
            print('WHAT?')
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
        x = self.conv_before_upsample(x) #new conv
        return self.conv_last(self.final_upsample(x))
