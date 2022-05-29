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


def get_mask_from_lengths(lengths, max_len=None):
    """Generate mask array from length.

    Args:
        lengths:
            A tensor of shape (b), where b is the batch size.

    Return:
        A mask tensor of shape (b,max_seq_len), where max_seq_len is the length
        of the longest sequence. Positions of padded elements will be set to
        True.
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = int(torch.max(lengths))

    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(
        batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
