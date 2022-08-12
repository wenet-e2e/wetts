# Copyright (c) 2021 Tsinghua University(Jie Chen)
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

import torch
from torch import nn

from wetts.utils import mask


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x: torch.Tensor, repeat_count: torch.Tensor):
        """Repeating all phonemes according to its duration.

        Args:
            x (torch.Tensor): Input phoneme sequences of shape (b,t_x,d)
            repeat_count (torch.Tensor): Duration of each phoneme of shape
            (b,t_x)
        """
        batch_size, input_max_seq_len = repeat_count.shape
        repeat_count = repeat_count.long()

        cum_duration = torch.cumsum(repeat_count, dim=1)  # (b,t_x)
        output_max_seq_len = torch.max(cum_duration)
        M = mask.get_content_mask(
            cum_duration.reshape(batch_size * input_max_seq_len),
            output_max_seq_len).reshape(
                batch_size, input_max_seq_len,
                output_max_seq_len).float()  # (b,t_x,t_y)
        M[:, 1:, :] = M[:, 1:, :] - M[:, :-1, :]
        return torch.bmm(M.permute(0, 2, 1), x), torch.max(cum_duration,
                                                           dim=1)[0]
