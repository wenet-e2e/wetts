# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#                    Tsinghua University(Jie Chen)
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

import torch
from torch import nn


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, repeat_count):
        batch_size, input_max_seq_len = repeat_count.shape
        output_seq_len = torch.sum(repeat_count, dim=-1)
        output_max_seq_len = int(torch.max(output_seq_len))
        M = torch.zeros([batch_size, output_max_seq_len, input_max_seq_len],
                        device=x.device)
        for i in range(batch_size):
            k = 0
            for j in range(input_max_seq_len):
                r = int(repeat_count[i, j])
                M[i, k:k + r, j] = 1
                k += r

        return torch.bmm(M, x), output_seq_len
