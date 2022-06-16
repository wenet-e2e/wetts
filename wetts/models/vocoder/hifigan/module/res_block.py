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

from torch import nn
import torch.nn.functional as F
from torch.nn import utils

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            utils.weight_norm(
                nn.Conv1d(channels,
                          channels,
                          kernel_size,
                          1,
                          dilation=dilation[0],
                          padding=get_padding(kernel_size, dilation[0]))),
            utils.weight_norm(
                nn.Conv1d(channels,
                          channels,
                          kernel_size,
                          1,
                          dilation=dilation[1],
                          padding=get_padding(kernel_size, dilation[1]))),
            utils.weight_norm(
                nn.Conv1d(channels,
                          channels,
                          kernel_size,
                          1,
                          dilation=dilation[2],
                          padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            utils.weight_norm(
                nn.Conv1d(channels,
                          channels,
                          kernel_size,
                          1,
                          dilation=1,
                          padding=get_padding(kernel_size, 1))),
            utils.weight_norm(
                nn.Conv1d(channels,
                          channels,
                          kernel_size,
                          1,
                          dilation=1,
                          padding=get_padding(kernel_size, 1))),
            utils.weight_norm(
                nn.Conv1d(channels,
                          channels,
                          kernel_size,
                          1,
                          dilation=1,
                          padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            utils.remove_weight_norm(l)
        for l in self.convs2:
            utils.remove_weight_norm(l)


class ResBlock2(nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            utils.weight_norm(
                nn.Conv1d(channels,
                          channels,
                          kernel_size,
                          1,
                          dilation=dilation[0],
                          padding=get_padding(kernel_size, dilation[0]))),
            utils.weight_norm(
                nn.Conv1d(channels,
                          channels,
                          kernel_size,
                          1,
                          dilation=dilation[1],
                          padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            utils.remove_weight_norm(l)
