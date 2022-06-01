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

from typing import Optional, Tuple

import torch
from torch import nn

from wetts.models.am.fastspeech2.module import variance_predictor
from wetts.models.am.fastspeech2.module import length_regulator


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, input_dim: int, n_conv_filter: int,
                 conv_kernel_size: int, dropout: float, pitch_min: float,
                 pitch_max: float, pitch_mean: float, pitch_sigma: float,
                 energy_min: float, energy_max: float, energy_mean: float,
                 energy_sigma: float, n_pitch_bin: int,
                 n_energy_bin: int) -> None:
        """Initializing variance adaptor.

        Args:
            input_dim (int): Dimension of input features.
            n_conv_filter (int): Number of convolution filters for each variance
            predictor.
            conv_kernel_size (int): Kernel size for all convolution layers in
            all variance predictors.
            dropout (float): Dropout of variance predictors.
            pitch_min (float): Minimum of pitch to construct pitch bins.
            pitch_max (float): Maximum of pitch to construct pitch bins.
            pitch_mean (float): Mean of pitch to construct pitch bins.
            pitch_sigma (float): Standard deviation of pitch to normalize
            minimum and maximum of pitch.
            energy_min (float): Minimum of energy to construct energy bins.
            energy_max (float): Maximum of energy to construct energy bins.
            energy_mean (float): Mean of energy to construct energy bins.
            energy_sigma (float): Standard deviation of energy to normalize
            minimum and maximum of energy.
            n_pitch_bin (int): Number of pitch bins for pitch quantization.
            n_energy_bin (int): Number of energy bins for energy quantization.
        """
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

    def _get_embedding(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            predictor: nn.Module,
            bins: nn.Parameter,
            embedding_table: nn.Module,
            target: Optional[torch.Tensor],
            control: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Getting pitch/energy predictions and embeddings.

        This function predicts pitch/energy from input features and generates
        pitch/energy embeddings. If pitch/energy target is given, it will be
        quantized by pitch/energy bins and converted to corresponding
        embeddings. Otherwise, predicted pitch/energy will be used to produced
        corresponding embeddings.

        Args:
            x (torch.Tensor): Input feature.
            x_mask (torch.Tensor): Mask for x. If x_mask[i,j] is True, x[i,j]
            will be set to zero in pitch/energy predictor.
            predictor (nn.Module): Pitch/energy predictor.
            bins (nn.Parameter): Pitch/energy bins for quantization.
            embedding_table (nn.Module): Pitch/energy embedding table to convert
            quantized value to embeddings.
            target (Optional[torch.Tensor]): Pitch/energy target.
            control (float, optional): Pitch/energy manipulation factor.
            Defaults to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prediction of pitch/energy,
            pitch/energy embeddings.
        """
        prediction = predictor(x, x_mask)
        if target is not None:
            embedding = embedding_table(torch.bucketize(target, bins))
        else:
            prediction *= control
            embedding = embedding_table(torch.bucketize(prediction, bins))
        return prediction, embedding

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        duration_target: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Predicting pitch, energy and duration and extending input phoneme
        sequences according to duration.

        Args:
            x (torch.Tensor): Input phoneme sequences.
            x_mask (torch.Tensor): Mask for x. If x_mask[i,j] is True, x[i,j]
            will be set to zero in variance predictor.
            duration_target (Optional[torch.Tensor], optional): Ground truth
            duration. Defaults to None.
            pitch_target (Optional[torch.Tensor], optional): Ground truth pitch.
            If this is not provided, the model will use predicted pitch.
            Defaults to None.
            energy_target (Optional[torch.Tensor], optional): Ground truth
            energy. If this is not provided, the model will use predicted
            energy. Defaults to None.
            p_control (float, optional): Pitch manipulation factor. Defaults to
            1.0.
            e_control (float, optional): Energy manipulation factor. Defaults to
            1.0.
            d_control (float, optional): Duration manipulation factor. Defaults
            to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor]: Output of variance adapter, mel-spectrogram length,
            pitch prediction, energy prediction and duration prediction in log
            domain.
        """

        log_duration_prediction = self.duration_predictor(x, x_mask)

        pitch_prediction, pitch_embedding = self._get_embedding(
            x, x_mask, self.pitch_predictor, self.pitch_bins,
            self.pitch_embedding, pitch_target, p_control)

        energy_prediction, energy_embedding = self._get_embedding(
            x + pitch_embedding, x_mask, self.energy_predictor,
            self.energy_bins, self.energy_embedding, energy_target, e_control)

        if duration_target is not None:
            output, mel_len = self.length_regulator(
                x + pitch_embedding + energy_embedding, duration_target)
        else:
            output, mel_len = self.length_regulator(
                x + pitch_embedding + energy_embedding,
                torch.round(torch.exp(log_duration_prediction)) * d_control)

        return (
            output,
            mel_len,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
        )
