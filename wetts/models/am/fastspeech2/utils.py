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
# Modified from FastSpeech2(https://github.com/ming024/FastSpeech2)

import torch
from torch import nn


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


def get_mask_from_lengths(lengths, max_len=None):
    """Generate mask array from length.

    Args: 
        lengths:
            A tensor of shape (b), where b is the batch size.
    
    Return: 
        A mask tensor of shape (b,max_seq_len), where max_seq_len is the length 
        of the longest sequence. Positions of padded elements will be set to True.
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = int(torch.max(lengths))

    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(
        batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def duration_to_log_duration(durations:torch.Tensor)->torch.Tensor:
    """Converting linear domain durations to log domain durations.

    Args:
        durations (torch.Tensor): Durations in linear domain.

    Returns:
        torch.Tensor: Log domain durations.
    """
    return torch.log(durations)
    

def log_duration_to_duration(log_durations:torch.Tensor)->torch.Tensor:
    """Converting log domain duration to linear domain duration.

    Args:
        log_durations (torch.Tensor): Durations in log domain.

    Returns:
        torch.Tensor: Linear domain durations.
    """
    return torch.round(torch.exp(log_durations))
