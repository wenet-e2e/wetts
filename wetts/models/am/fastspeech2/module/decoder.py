# Copyright (c) 2022 Horizon Robtics. (authors: Jie Chen)
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
# Modified from PaddleSpeech(https://github.com/PaddlePaddle/PaddleSpeech)
# and FastSpeech2(https://github.com/ming024/FastSpeech2)

from torch import nn

from wetts.models.am.fastspeech2.module import fft


class FastSpeech2Decoder(nn.Module):
    """FastSpeech2 Decoder.
    This is an implementation of decoder for FastSpeech2.
    """

    def __init__(self, dec_hidden_dim, n_dec_layer, n_dec_head,
                 n_dec_conv_filter, dec_conv_kernel_size, dec_dropout):
        super(FastSpeech2Decoder, self).__init__()

        self.dec_hidden_dim = dec_hidden_dim

        self.layer_stack = nn.ModuleList([
            fft.FFTBlock(dec_hidden_dim, n_dec_head,
                         dec_hidden_dim // n_dec_head,
                         dec_hidden_dim // n_dec_head, n_dec_conv_filter,
                         dec_conv_kernel_size, dec_dropout)
            for _ in range(n_dec_layer)
        ])

    def forward(self, x, padding_mask):
        """

        Args:
            x: A tensor of shape (b,t',d)
            padding_mask: A BoolTensor of shape (b,t'). If padding_mask[i,j] is
            True, x[i,j,:] will be masked in self-attention.
        Returns:
            decoder output, decoder attention
        """
        dec_slf_attn_list = []
        max_seq_len = x.shape[1]
        slf_attn_mask = padding_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        for dec_layer in self.layer_stack:
            x, dec_slf_attn = dec_layer(x,
                                        mask=padding_mask,
                                        slf_attn_mask=slf_attn_mask)
            dec_slf_attn_list += [dec_slf_attn]
        return x, dec_slf_attn_list

    def inference(self, x, padding_mask):
        return self.forward(x, padding_mask)
