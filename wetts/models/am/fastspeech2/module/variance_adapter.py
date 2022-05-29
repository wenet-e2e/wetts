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
import math

from wetts.models.am.fastspeech2.module import variance_predictor
from wetts.models.am.fastspeech2.module import length_regulator
from wetts.models.am.fastspeech2 import utils


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, input_dim, n_conv_filter, conv_kernel_size, dropout,
                 pitch_min, pitch_max, pitch_mean, pitch_sigma, energy_min,
                 energy_max, energy_mean, energy_sigma, n_pitch_bin,
                 n_energy_bin):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = variance_predictor.VariancePredictor(
            input_dim, n_conv_filter, conv_kernel_size, dropout)
        self.pitch_predictor = variance_predictor.VariancePredictor(
            input_dim, n_conv_filter, conv_kernel_size, dropout)
        self.energy_predictor = variance_predictor.VariancePredictor(
            input_dim, n_conv_filter, conv_kernel_size, dropout)
        self.length_regulator = length_regulator.LengthRegulator()

        self.pitch_bins = nn.Parameter(
            torch.linspace((pitch_min - pitch_mean) / pitch_sigma,
                           (pitch_max - pitch_mean) / pitch_sigma,
                           n_pitch_bin - 1),
            requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace((energy_min - energy_mean) / energy_sigma,
                           (energy_max - energy_mean) / energy_sigma,
                           n_energy_bin - 1),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(n_pitch_bin, input_dim)
        self.energy_embedding = nn.Embedding(n_energy_bin, input_dim)

    def get_pitch_embedding(self, pitch):
        return self.pitch_embedding(torch.bucketize(pitch, self.pitch_bins))

    def get_energy_embedding(self, energy):
        return self.energy_embedding(torch.bucketize(energy, self.energy_bins))

    def forward(
        self,
        x,
        x_mask,
        duration_target,
        pitch_target,
        energy_target,
    ):
        """

        Args:
            x: Input phoneme sequence.
            x_mask: Mask tensor for x. If x_mask[i,j] is True, x will be masked
            in self-attention.
            duration_target: Ground truth duration.
            pitch_target: Ground truth pitch.
            energy_target: Ground truth energy.
        """

        log_duration_prediction = self.duration_predictor(x, x_mask)

        pitch_prediction = self.pitch_predictor(x, x_mask)
        pitch_target_embedding = self.get_pitch_embedding(pitch_target)

        # teacher forcing
        energy_prediction = self.energy_predictor(x + pitch_target_embedding,
                                                  x_mask)
        energy_target_embedding = self.get_energy_embedding(energy_target)

        # teacher forcing
        output, mel_len = self.length_regulator(
            x + pitch_target_embedding + energy_target_embedding,
            duration_target)

        return (
            output,
            mel_len,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
        )

    def inference(self,
                  x,
                  x_mask,
                  p_control=1.0,
                  e_control=1.0,
                  d_control=1.0):
        """

        Args:
            x: Input phoneme sequence.
            x_mask: Mask tensor for x. If x_mask[i,j] is True, x will be masked
            in self-attention.
            p_control: Pitch manipulation factor.
            e_control: Energy manipulation factor.
            d_control: Duration manipulation factor.
        """
        log_duration_prediction = self.duration_predictor(x, x_mask)
        predicted_duration = utils.log_duration_to_duration(
            log_duration_prediction) * d_control

        pitch_prediction = self.pitch_predictor(x, x_mask)
        pitch_prediction_embedding = self.get_pitch_embedding(
            pitch_prediction) * p_control

        energy_prediction = self.energy_predictor(
            x + pitch_prediction_embedding, x_mask)
        energy_prediction_embedding = self.get_energy_embedding(
            energy_prediction) * e_control

        output, mel_len = self.length_regulator(
            x + pitch_prediction_embedding + energy_prediction_embedding,
            predicted_duration)
        return output, mel_len