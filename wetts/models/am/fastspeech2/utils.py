# Copyright (c) 2022 Tsinghua University(Jie Chen).
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


def get_sinusoid_encoding_table(max_seq_len, input_dim, padding_idx=None):
    """Sinusoid position encoding table"""

    def cal_angle(position, idx):
        return position / 10000**(2 * (idx // 2) / input_dim)

    def get_posi_angle_vec(position):
        return [cal_angle(position, idx) for idx in range(input_dim)]

    sinusoid_table = torch.tensor(
        [get_posi_angle_vec(pos_i) for pos_i in range(max_seq_len)])

    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)
