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

import torch
from torch import nn


class FastSpeech2Loss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.masked_mse_loss = MaskedLoss(nn.MSELoss)
        self.masked_l1_loss = MaskedLoss(nn.L1Loss)

    def forward(self, duration_target, duration_prediction, pitch_target,
                pitch_prediction, energy_target, energy_prediction,
                variance_mask, mel_target, mel_prediction,
                postnet_mel_prediction, mel_mask):

        return (self.masked_mse_loss(duration_prediction, duration_target,
                                     variance_mask),
                self.masked_mse_loss(pitch_prediction, pitch_target,
                                     variance_mask),
                self.masked_mse_loss(energy_prediction, energy_target,
                                     variance_mask),
                self.masked_l1_loss(mel_prediction, mel_target, mel_mask),
                self.masked_l1_loss(postnet_mel_prediction, mel_target,
                                    mel_mask))


class MaskedLoss(nn.Module):

    def __init__(self, LossClass) -> None:
        super().__init__()
        self.loss = LossClass()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Calculating masked loss.

        Args:
            prediction (torch.Tensor): Prediction.
            target (torch.Tensor): Target.
            mask (torch.Tensor): Mask tensor, where 1 indicates here is a
            padded element.

        Returns:
            torch.Tensor: Loss between prediction and target.
        """
        return self.loss(
            prediction.masked_select(~mask).float(),
            target.masked_select(~mask).float())
