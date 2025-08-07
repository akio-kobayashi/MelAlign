import torch
import torch.nn as nn

def CausalConv1d(in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 stride: int = 1,
                 **kwargs) -> nn.Conv1d:
    """
    1D causal convolution: pads input so output has same length.
    """
    padding = (kernel_size - 1) * dilation
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        **kwargs
    )


class ConvBlock(nn.Module):
    """
    A single causal Conv1d block with LayerNorm, activation, dropout.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 stride: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        length = x.size(1)
        x = self.norm(x)
        # to [B, C, T]
        x = x.transpose(1, 2)
        x = self.conv(self.activation(x))
        x = self.dropout(x)
        # trim to original length and back to [B, T, F]
        x = x[:, :, :length]
        x = x.transpose(1, 2)
        return x


class ConvStack(nn.Module):
    """
    Stack of ConvBlocks defined by channel progression.
    """

    def __init__(self,
                 channels: list[int],
                 kernel_size: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.append(
                ConvBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNet(nn.Module):
    """
    Front-end conv stack projecting from mel_dim to model_dim.
    """

    def __init__(self,
                 mel_dim: int = 80,
                 model_dim: int = 512,
                 hidden_dims: list[int] | None = None,
                 kernel_size: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        dims = [mel_dim] + (hidden_dims or [512, 512]) + [model_dim]
        self.stack = ConvStack(dims, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class Upsampler(nn.Module):
    """
    Upsample along time dimension by scale factor (int or tuple).
    """

    def __init__(self,
                 input_dim: int,
                 scale_factor: int | tuple[int, int]):
        super().__init__()
        self.input_dim = input_dim
        self.upsampler = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C*F] -> [B, C, T, F] without einops
        b, t, cf = x.shape
        f = self.input_dim
        c = cf // f
        # reshape to [b, t, c, f]
        x = x.view(b, t, c, f)
        # permute to [b, c, t, f]
        x = x.permute(0, 2, 1, 3)
        x = self.upsampler(x)
        # back to [b, t', c, f]
        x = x.permute(0, 2, 1, 3)
        # flatten to [b, t', c*f]
        x = x.reshape(b, x.size(1), c * f)        
        return x


class PostNet(nn.Module):
    """
    Back-end conv stack projecting from model_dim back to mel_dim.
    """

    def __init__(self,
                 model_dim: int = 512,
                 mel_dim: int = 80,
                 hidden_dims: list[int] | None = None,
                 kernel_size: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        dims = [model_dim] + (hidden_dims or [512, 512]) + [mel_dim]
        self.stack = ConvStack(dims, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
