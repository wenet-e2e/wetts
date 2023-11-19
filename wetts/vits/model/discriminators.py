import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm

from model.modules import LRELU_SLOPE
from model.normalization import LayerNorm
from utils.commons import get_padding

AVAILABLE_DURATION_DISCRIMINATOR_TYPES = [
    "dur_disc_1",
    "dur_disc_2",
]


class DiscriminatorP(nn.Module):

    def __init__(self,
                 period,
                 kernel_size=5,
                 stride=3,
                 use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(
                    1,
                    32,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )),
            norm_f(
                Conv2d(
                    32,
                    128,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )),
            norm_f(
                Conv2d(
                    128,
                    512,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )),
            norm_f(
                Conv2d(
                    512,
                    1024,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )),
            norm_f(
                Conv2d(
                    1024,
                    1024,
                    (kernel_size, 1),
                    1,
                    padding=(get_padding(kernel_size, 1), 0),
                )),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
            for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DurationDiscriminatorV1(nn.Module):  # vits2
    # TODO : not using "spk conditioning" for now according to the paper.
    # Can be a better discriminator if we use it.
    def __init__(self,
                 in_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout,
                 gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        # self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        # self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(2 * filter_channels,
                                        filter_channels,
                                        kernel_size,
                                        padding=kernel_size // 2)
        self.pre_out_norm_1 = LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(filter_channels,
                                        filter_channels,
                                        kernel_size,
                                        padding=kernel_size // 2)
        self.pre_out_norm_2 = LayerNorm(filter_channels)

        # if gin_channels != 0:
        #   self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1),
                                          nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        # x = torch.relu(x)
        # x = self.pre_out_norm_1(x)
        # x = self.drop(x)
        x = self.pre_out_conv_2(x * x_mask)
        # x = torch.relu(x)
        # x = self.pre_out_norm_2(x)
        # x = self.drop(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        # if g is not None:
        #   g = torch.detach(g)
        #   x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        # x = torch.relu(x)
        # x = self.norm_1(x)
        # x = self.drop(x)
        x = self.conv_2(x * x_mask)
        # x = torch.relu(x)
        # x = self.norm_2(x)
        # x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)

        return output_probs


class DurationDiscriminatorV2(nn.Module):  # vits2
    # TODO : not using "spk conditioning" for now according to the paper.
    # Can be a better discriminator if we use it.
    def __init__(self,
                 in_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout,
                 gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.conv_1 = nn.Conv1d(in_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(2 * filter_channels,
                                        filter_channels,
                                        kernel_size,
                                        padding=kernel_size // 2)
        self.pre_out_norm_1 = LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(filter_channels,
                                        filter_channels,
                                        kernel_size,
                                        padding=kernel_size // 2)
        self.pre_out_norm_2 = LayerNorm(filter_channels)

        # if gin_channels != 0:
        #   self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1),
                                          nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        # if g is not None:
        #   g = torch.detach(g)
        #   x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append([output_prob])

        return output_probs
