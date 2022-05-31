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
# Modified from FastSpeech2(https://github.com/ming024/FastSpeech2)

import collections

from torch import nn


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, input_dim, n_conv_filter, conv_kernel_size, dropout):
        super(VariancePredictor, self).__init__()
        self.input_dim = input_dim
        self.n_conv_filter = n_conv_filter
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(
            collections.OrderedDict([
                (
                    "conv1d_1",
                    Conv(
                        self.input_dim,
                        self.n_conv_filter,
                        kernel_size=self.conv_kernel_size,
                    ),
                ),
                ("relu_1", nn.ReLU()),
                ("layer_norm_1", nn.LayerNorm(self.n_conv_filter)),
                ("dropout_1", nn.Dropout(self.dropout)),
                (
                    "conv1d_2",
                    Conv(
                        self.n_conv_filter,
                        self.n_conv_filter,
                        kernel_size=self.conv_kernel_size,
                    ),
                ),
                ("relu_2", nn.ReLU()),
                ("layer_norm_2", nn.LayerNorm(self.n_conv_filter)),
                ("dropout_2", nn.Dropout(self.dropout)),
            ]))

        self.linear_layer = nn.Linear(self.n_conv_filter, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
    ):
        """

        Args:
            in_channels: dimension of input
            out_channels: dimension of output
            kernel_size: size of kernel
            bias: boolean. if True, bias is included.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding='same',
            bias=bias,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        return x
