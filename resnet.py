from typing import List, Optional

import torch
from torch import nn

class Shortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        # in_channels is the number of channels in x
        # out_channels is the number of channels in F(x,{Wi​}) i.e. y 
        # stride is the stride length in the convolution operation for F. We do the same stride on the shortcut connection, to match the feature-map size.
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        # convolution layer with kernel size 1x1, stride as specified, and no bias term
        self.bn = nn.BatchNorm2d(out_channels)  # batch normalization layer for the output of the convolution
    def forward(self, x: torch.Tensor):
        # x is the input tensor
        # apply the convolution and batch normalization to x
        return self.bn(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output channels are different or the stride is not 1,
        # we need to use a shortcut connection to match the dimensions.
        if in_channels != out_channels or stride != 1:
            self.shortcut = Shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()  # No change needed for the shortcut
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)  # Apply the shortcut connection
        x = self.act1(self.bn1(self.conv1(x)))  # First convolution, batch norm, and activation
        x = self.bn2(self.conv2(x))  # Second convolution and batch norm
        return self.act2(x + shortcut)  # Add the shortcut and apply the final activation

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # If the input and output channels are different or the stride is not 1,
        # we need to use a shortcut connection to match the dimensions.
        if in_channels != out_channels or stride != 1:
            self.shortcut = Shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()  # No change needed for the shortcut
        self.act3 = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)  # Apply the shortcut connection
        x = self.act1(self.bn1(self.conv1(x)))  # First convolution, batch norm, and activation
        x = self.act2(self.bn2(self.conv2(x)))  # Second convolution, batch norm, and activation
        x = self.bn3(self.conv3(x))  # Third convolution and batch norm
        return self.act3(x + shortcut)  # Add the shortcut and apply the final activation

class ResNet(nn.Module):
    def __init__(self, n_blocks: List[int], n_channels: List[int],
                 bottlenecks: Optional[List[int]] = None,
                 img_channels: int = 3, first_kernel_size: int = 7):
        """
        * `n_blocks` is a list of of number of blocks for each feature map size.
        * `n_channels` is the number of channels for each feature map size.
        * `bottlenecks` is the number of channels the bottlenecks.
        If this is `None`, [residual blocks](#residual_block) are used.
        * `img_channels` is the number of channels in the input.
        * `first_kernel_size` is the kernel size of the initial convolution layer
        """
        super().__init__()

        # Number of blocks and number of channels for each feature map size
        assert len(n_blocks) == len(n_channels)
        # If [bottleneck residual blocks](#bottleneck_residual_block) are used,
        # the number of channels in bottlenecks should be provided for each feature map size
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        # Initial convolution layer maps from `img_channels` to number of channels in the first
        # residual block (`n_channels[0]`)
        self.conv = nn.Conv2d(img_channels, n_channels[0],
                              kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        # Batch norm after initial convolution
        self.bn = nn.BatchNorm2d(n_channels[0])

        # List of blocks
        blocks = []
        # Number of channels from previous layer (or block)
        prev_channels = n_channels[0]
        # Loop through each feature map size
        for i, channels in enumerate(n_channels):
            # The first block for the new feature map size, will have a stride length of $2$
            # except fro the very first block
            stride = 2 if len(blocks) == 0 else 1

            if bottlenecks is None:
                # [residual blocks](#residual_block) that maps from `prev_channels` to `channels`
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                # [bottleneck residual blocks](#bottleneck_residual_block)
                # that maps from `prev_channels` to `channels`
                blocks.append(BottleneckResidualBlock(prev_channels, bottlenecks[i], channels,
                                                      stride=stride))

            # Change the number of channels
            prev_channels = channels
            # Add rest of the blocks - no change in feature map size or channels
            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    # [residual blocks](#residual_block)
                    blocks.append(ResidualBlock(channels, channels, stride=1))
                else:
                    # [bottleneck residual blocks](#bottleneck_residual_block)
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, stride=1))

        # Stack the blocks
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, img_channels, height, width]`
        """

        # Initial convolution and batch normalization
        x = self.bn(self.conv(x))
        # Residual (or bottleneck) blocks
        x = self.blocks(x)
        # Change `x` from shape `[batch_size, channels, h, w]` to `[batch_size, channels, h * w]`
        x = x.view(x.shape[0], x.shape[1], -1)
        # Global average pooling
        return x.mean(dim=-1)