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

import torch
from torch import nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).reshape(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).reshape(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).reshape(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).reshape(sz_b, len_q,
                                                    -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, input_dim, n_conv_filter, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            input_dim,
            n_conv_filter,
            kernel_size=kernel_size[0],
            padding='same',
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            n_conv_filter,
            input_dim,
            kernel_size=kernel_size[1],
            padding='same',
        )

        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (b,t,d)
        """
        residual = x
        output = x.transpose(1, 2)
        output = F.relu(self.w_2(F.relu(self.w_1(output))))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 n_head,
                 d_k,
                 d_v,
                 n_conv_filter,
                 kernel_size,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head,
                                           d_model,
                                           d_k,
                                           d_v,
                                           dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model,
                                               n_conv_filter,
                                               kernel_size,
                                               dropout=dropout)

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input,
                                                 enc_input,
                                                 enc_input,
                                                 mask=slf_attn_mask)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn
