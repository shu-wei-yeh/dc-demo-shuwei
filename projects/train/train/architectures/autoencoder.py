from typing import Optional

import torch.nn as nn

from train.architectures import Architecture


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        transpose: bool,
        output_padding: Optional[int] = None,
        activation: nn.Module = nn.Tanh,
    ) -> None:
        super().__init__()

        if not transpose and output_padding is not None:
            raise ValueError(
                "Cannot specify output padding to non-transposed convolution"
            )
        elif output_padding is not None:
            kwargs = {"output_padding": output_padding}
        else:
            kwargs = {}

        conv_op = nn.ConvTranspose1d if transpose else nn.Conv1d
        self.conv = conv_op(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Autoencoder(Architecture):
    def __init__(
        self,
        num_witnesses: int,
        hidden_channels: list[int],
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()

        self.num_witnesses = num_witnesses
        self.input_conv = ConvBlock(
            num_witnesses,
            num_witnesses,
            kernel_size=7,
            stride=1,
            padding=3,
            transpose=False,
            activation=activation,
        )

        self.downsampler = nn.Sequential()
        in_channels = num_witnesses
        for i, out_channels in enumerate(hidden_channels):
            conv_block = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                transpose=False,
                activation=activation,
            )
            self.downsampler.add_module(f"CONV_{i+1}", conv_block)
            in_channels = out_channels

        self.upsampler = nn.Sequential()
        out_layers = hidden_channels[-2:None:-1] + [1]
        for i, out_channels in enumerate(out_layers):
            conv_block = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
                transpose=True,
                activation=activation,
            )
            self.upsampler.add_module(f"CONVTRANS_{i+1}", conv_block)
            in_channels = out_channels

        self.output_conv = nn.Conv1d(
            in_channels, 1, kernel_size=7, stride=1, padding=3
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.downsampler(x)
        x = self.upsampler(x)
        x = self.output_conv(x)
        return x[:, 0]
