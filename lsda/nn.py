r"""Neural networks"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import *


class EncoderDecoder(nn.Module):
    def __init__(
        self, 
        in_features: int,
        hidden_features: int,
        out_features: int,
        window: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.window = window
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            activation(),
            nn.Linear(hidden_features, hidden_features)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            activation(),
            nn.Linear(hidden_features, out_features)
        )

        
class ConvEncoderDecoder(nn.Module):
    def __init__(
        self, 
        in_features: int,
        hidden_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        aux_features: int = 0,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.in_features = in_features
        self.forward_channels = (in_features + 1, ) + hidden_channels
        self.reverse_channels = hidden_channels[::-1] + (in_features, ) 
        self.kernel_sizes = kernel_sizes
        self.encoder = list()
        self.decoder = list()
        
        for n_layer in range(len(hidden_channels)):
            self.encoder.append(
                nn.Conv2d(self.forward_channels[n_layer], 
                          self.forward_channels[n_layer + 1], 
                          self.kernel_sizes[n_layer], 1, (self.kernel_sizes[n_layer] - 1) // 2)
            )
            
            self.decoder.append(
                nn.ConvTranspose2d(self.reverse_channels[n_layer], 
                                   self.reverse_channels[n_layer + 1], 
                                   self.kernel_sizes[n_layer], 1, (self.kernel_sizes[n_layer] - 1) // 2)
            )
            
            if n_layer < len(hidden_channels) - 1:
                self.encoder.append(nn.BatchNorm2d(self.forward_channels[n_layer + 1]))
                self.decoder.append(nn.BatchNorm2d(self.reverse_channels[n_layer + 1]))
                self.encoder.append(activation())
                self.decoder.append(activation())
                
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)


class ResidualBlock(nn.Sequential):
    r"""Creates a residual block."""

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ModResidualBlock(nn.Module):
    r"""Creates a residual block with modulation."""

    def __init__(self, project: nn.Module, residue: nn.Module):
        super().__init__()

        self.project = project
        self.residue = residue

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + self.residue(x + self.project(y))


class ResMLP(nn.Sequential):
    r"""Creates a residual multi-layer perceptron (ResMLP).

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The number of hidden features.
        activation: The activation function constructor.
        kwargs: Keyword arguments passed to :class:`nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.ReLU,
        **kwargs,
    ):
        blocks = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            if after != before:
                blocks.append(nn.Linear(before, after, **kwargs))

            blocks.append(
                ResidualBlock(
                    nn.LayerNorm(after),
                    nn.Linear(after, after, **kwargs),
                    activation(),
                    nn.Linear(after, after, **kwargs),
                )
            )

        super().__init__(*blocks)

        self.in_features = in_features
        self.out_features = out_features
        

class UNet(nn.Module):
    r"""Creates a U-Net with modulation.

    References:
        | U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        | https://arxiv.org/abs/1505.04597

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        mod_features: The number of modulation features.
        hidden_channels: The number of hidden channels.
        hidden_blocks: The number of hidden blocks at each depth.
        kernel_size: The size of the convolution kernels.
        stride: The stride of the downsampling convolutions.
        activation: The activation function constructor.
        spatial: The number of spatial dimensions. Can be either 1, 2 or 3.
        kwargs: Keyword arguments passed to :class:`nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hidden_channels: Sequence[int] = (32, 64, 128),
        hidden_blocks: Sequence[int] = (2, 3, 5),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        activation: Callable[[], nn.Module] = nn.ReLU,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial = spatial

        # Components
        convolution = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }.get(spatial)

        if type(kernel_size) is int:
            kernel_size = [kernel_size] * spatial

        if type(stride) is int:
            stride = [stride] * spatial

        kwargs.update(
            kernel_size=kernel_size,
            padding=[k // 2 for k in kernel_size],
        )

        block = lambda channels: ModResidualBlock(
            project=nn.Sequential(
                nn.Linear(mod_features, channels),
                nn.Unflatten(-1, (-1,) + (1,) * spatial),
            ),
            residue=nn.Sequential(
                convolution(channels, channels, **kwargs),
                activation(),
                convolution(channels, channels, **kwargs),
            ),
        )

        # Layers
        heads, tails = [], []
        descent, ascent = [], []

        for i, blocks in enumerate(hidden_blocks):
            if i > 0:
                heads.append(
                    nn.Sequential(
                        convolution(
                            hidden_channels[i - 1],
                            hidden_channels[i],
                            stride=stride,
                            **kwargs,
                        ),
                    )
                )

                tails.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=tuple(stride), mode='nearest'),
                        convolution(
                            hidden_channels[i],
                            hidden_channels[i - 1],
                            **kwargs,
                        ),
                    )
                )
            else:
                heads.append(convolution(in_channels, hidden_channels[i], **kwargs))
                tails.append(convolution(hidden_channels[i], out_channels, **kwargs))

            descent.append(nn.ModuleList(block(hidden_channels[i]) for _ in range(blocks)))
            ascent.append(nn.ModuleList(block(hidden_channels[i]) for _ in range(blocks)))

        self.heads = nn.ModuleList(heads)
        self.tails = nn.ModuleList(reversed(tails))
        self.descent = nn.ModuleList(descent)
        self.ascent = nn.ModuleList(reversed(ascent))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        memory = []

        for head, blocks in zip(self.heads, self.descent):
            x = head(x)

            for block in blocks:
                x = block(x, y)

            memory.append(x)
            
        memory.pop()

        for blocks, tail in zip(self.ascent, self.tails):
            for block in blocks:
                x = block(x, y)
            if memory:
                x = tail(x) + memory.pop()
            else:
                x = tail(x)

        return x
    
    
class GANResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(GANResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)
