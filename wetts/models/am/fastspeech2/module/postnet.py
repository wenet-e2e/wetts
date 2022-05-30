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
# Modified from Tacotron2(https://github.com/NVIDIA/tacotron2)

import torch
from torch import nn


class ConvNorm(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 bias=True,
                 w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    padding='same',
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, mel_dim, kernel_size, hidden_dim, n_layers, dropout):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(mel_dim,
                         hidden_dim,
                         kernel_size=kernel_size,
                         w_init_gain='tanh'), nn.BatchNorm1d(hidden_dim),
                nn.Tanh(), nn.Dropout(dropout)))

        for i in range(1, n_layers - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hidden_dim,
                             hidden_dim,
                             kernel_size=kernel_size,
                             w_init_gain='tanh'), nn.BatchNorm1d(hidden_dim),
                    nn.Tanh(), nn.Dropout(dropout)))

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hidden_dim,
                         mel_dim,
                         kernel_size=kernel_size,
                         w_init_gain='linear'), nn.BatchNorm1d(mel_dim),
                nn.Dropout(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for postnet.

        Args:
            x (torch.Tensor): Input mel of shape (b,T,C).

        Returns:
            torch.Tensor: output mel.
        """
        x = x.permute(0, 2, 1)
        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)
        x = self.convolutions[-1](x)

        return x.permute(0, 2, 1)
