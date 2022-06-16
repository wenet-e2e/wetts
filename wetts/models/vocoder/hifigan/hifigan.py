# Copyright (c) 2022 Tsinghua University(Jie Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from HiFi-GAN(https://github.com/jik876/hifi-gan)

from typing import List

import torch
from torch import nn
from torch.nn import utils
import torch.nn.functional as F

from wetts.models.vocoder.hifigan.module import res_block
from wetts.models.vocoder.hifigan.module import discriminator


class Generator(nn.Module):
    """HiFiGAN generator.

    """
    def __init__(self,
                 resblock_kernel_sizes: List[int],
                 resblock_dilation_sizes: List[List[int]],
                 upsample_rates: List[int],
                 upsample_kernel_sizes: List[int],
                 upsample_initial_channel: int,
                 resblock_type: str = 'type1'):
        """Initializing a HiFiGAN generator.

        Args:
            resblock_kernel_sizes (List[int]): Kernel sizes of resblock.
            resblock_dilation_sizes (List[List[int]]): Dilation of resblock.
            upsample_rates (List[int]): Upsampling rates for generator.
            upsample_kernel_sizes (List[int]): Kernel sizes for upsampling.
            upsample_initial_channel (int): Number of channels for initial
            upsampling layer.
            resblock_type (str, optional): Type of resblock to use. Defaults to
            'type1'.
        """
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = utils.weight_norm(
            nn.Conv1d(80, upsample_initial_channel, 7, 1, padding=3))
        resblock = (res_block.ResBlock1
                    if resblock_type == 'type1' else res_block.ResBlock2)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                utils.weight_norm(
                    nn.ConvTranspose1d(upsample_initial_channel // (2**i),
                                       upsample_initial_channel //
                                       (2**(i + 1)),
                                       k,
                                       u,
                                       padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(res_block.init_weights)
        self.conv_post.apply(res_block.init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, res_block.LRELU_SLOPE)
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
        print('Removing weight norm...')
        for l in self.ups:
            utils.remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        utils.remove_weight_norm(self.conv_pre)
        utils.remove_weight_norm(self.conv_post)


class MultiPeriodDiscriminator(nn.Module):

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            discriminator.DiscriminatorP(2),
            discriminator.DiscriminatorP(3),
            discriminator.DiscriminatorP(5),
            discriminator.DiscriminatorP(7),
            discriminator.DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiScaleDiscriminator(nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            discriminator.DiscriminatorS(use_spectral_norm=True),
            discriminator.DiscriminatorS(),
            discriminator.DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2),
             nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
