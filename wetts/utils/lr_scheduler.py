# Copyright (c) 2022 Tsinghua University (Jie Chen)
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

from torch.optim.lr_scheduler import LambdaLR


def transformer_lr_scheduler(optimizer, d_model: int, warmup_steps: int):
    """Transformer learning rate scheduler.

    Args:
        optimizer: Optimizer. The learning rate of this optimizer should be set
        to 1.
        d_model (int): d_model for learning rate calculation.
        warmup_steps (int): warmup steps for learning rate calculation.
    """

    def lr():

        def get_lr(step):
            step = max(step, 1)
            return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)

        return get_lr

    return LambdaLR(optimizer, lr())
