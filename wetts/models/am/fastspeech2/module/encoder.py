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

from torch import nn

from wetts.models.am.fastspeech2.module import fft
from wetts.models.am.fastspeech2.module import length_regulator


class FastSpeech2Encoder(nn.Module):
    """FastSpeech2 Encoder.
    This is an implementation of encoder for FastSpeech2.
    """

    def __init__(self, enc_hidden_dim, n_enc_layer, n_enc_head,
                 n_enc_conv_filter, enc_conv_kernel_size, enc_dropout):
        super().__init__()

        self.enc_hidden_dim = enc_hidden_dim

        # Length regulator are used here to remove tokens of type 0, which
        # are prosodic structure labels in mandarin.
        self.length_regulator = length_regulator.LengthRegulator()

        self.layer_stack = nn.ModuleList([
            fft.FFTBlock(enc_hidden_dim, n_enc_head,
                         enc_hidden_dim // n_enc_head,
                         enc_hidden_dim // n_enc_head, n_enc_conv_filter,
                         enc_conv_kernel_size, enc_dropout)
            for _ in range(n_enc_layer)
        ])

    def forward(self, x, padding_mask, token_type):
        """

        Args:
            x: A tensor of shape (b,t',d)
            padding_mask: A BoolTensor of shape (b,t'). If padding_mask[i,j] is
            True, x[i,j,:] will be masked in self-attention.
            token_type: An IntTensor of shape (b,t). Assume the output of the
            final FFT block is final_FFT_output and the final output of encoder
            is enc_output. When token_type[i,j] is 0, final_FFT_output[i,j,:]
            will be removed from enc_output.
        Returns:
            encoder output, the length of encoder output, encoder attention
        """
        enc_slf_attn_list = []
        max_seq_len = x.shape[1]
        slf_attn_mask = padding_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        for enc_layer in self.layer_stack:
            x, enc_slf_attn = enc_layer(x,
                                        mask=padding_mask,
                                        slf_attn_mask=slf_attn_mask)
            enc_slf_attn_list += [enc_slf_attn]
        x, seq_len = self.length_regulator(x, token_type)
        return x, seq_len, enc_slf_attn_list
