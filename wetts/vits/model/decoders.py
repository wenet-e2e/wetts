import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import InverseSpectrogram

from model.modules import LRELU_SLOPE
from model.normalization import LayerNorm
from utils.commons import init_weights, get_padding
from utils.stft import OnnxSTFT


class Generator(nn.Module):

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel,
                               upsample_initial_channel,
                               7,
                               1,
                               padding=3)
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2**(i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class ResBlock1(torch.nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                )),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                )),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                )),
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class ConvNeXtLayer(nn.Module):

    def __init__(self, channels, h_channels, scale):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.norm = LayerNorm(channels)
        self.pw_conv1 = nn.Conv1d(channels, h_channels, 1)
        self.pw_conv2 = nn.Conv1d(h_channels, channels, 1)
        self.scale = nn.Parameter(torch.full(size=(1, channels, 1),
                                             fill_value=scale),
                                  requires_grad=True)

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = res + x
        return x


class VocosGenerator(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 h_channels,
                 out_channels,
                 num_layers,
                 istft_config,
                 gin_channels,
                 is_onnx=False):
        super().__init__()

        self.pad = nn.ReflectionPad1d([1, 0])
        self.in_conv = nn.Conv1d(in_channels,
                                 channels,
                                 kernel_size=1,
                                 padding=0)
        self.cond = Conv1d(gin_channels, channels, 1)
        self.norm_pre = LayerNorm(channels)
        scale = 1 / num_layers
        self.layers = nn.ModuleList([
            ConvNeXtLayer(channels, h_channels, scale)
            for _ in range(num_layers)
        ])
        self.norm_post = LayerNorm(channels)
        self.out_conv = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.is_onnx = is_onnx

        if self.is_onnx:
            self.stft = OnnxSTFT(filter_length=istft_config['n_fft'],
                                 hop_length=istft_config['hop_length'],
                                 win_length=istft_config['win_length'])
        else:
            self.istft = InverseSpectrogram(**istft_config)

    def forward(self, x, g=None):
        x = self.pad(x)
        x = self.in_conv(x) + self.cond(g)
        x = self.norm_pre(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_post(x)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp_max(max=1e2)
        if self.is_onnx:
            o = self.stft.inverse(mag, phase).to(x.device)
        else:
            s = mag * (phase.cos() + 1j * phase.sin())
            o = self.istft(s).unsqueeze(1)
        return o

    def remove_weight_norm(self):
        pass
