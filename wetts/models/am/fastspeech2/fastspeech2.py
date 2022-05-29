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
# Modified from PaddleSpeech(https://github.com/PaddlePaddle/PaddleSpeech)
# and FastSpeech2(https://github.com/ming024/FastSpeech2)

from typing import List, Tuple

import torch
from torch import nn

from wetts.models.am.fastspeech2.module import encoder
from wetts.models.am.fastspeech2.module import variance_adapter
from wetts.models.am.fastspeech2.module import decoder
from wetts.models.am.fastspeech2 import utils
from wetts.models.am.fastspeech2.module import postnet


class FastSpeech2(nn.Module):
    """FastSpeech2 module.
    This is an implementation of FastSpeech2 described in `FastSpeech 2: Fast
    and High-Quality End-to-End Text to Speech`_.
    Phoneme level pitch and energy predictor are used in Variance Adapter. Pitch
    and energy are quantized and are converted into corresponding embeddings.
    For mandarian, explicit prosody control is implemented by inserting prosodic
    structure labels into input phoneme sequences. Embeddings of prosodic
    structure labels are removed after encoder.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558
    .. _`FastPitch: Parallel Text-to-speech with Pitch Prediction`:
        https://arxiv.org/abs/2006.06873
    """

    def __init__(self, enc_hidden_dim: int, n_enc_layer: int, n_enc_head: int,
                 n_enc_conv_filter: int, enc_conv_kernel_size: List[int],
                 enc_dropout: float, n_vocab: int, padding_idx: int,
                 n_va_conv_filter: int, va_conv_kernel_size: int,
                 va_dropout: float, pitch_min: float, pitch_max: float,
                 pitch_mean: float, pitch_sigma: float, energy_min: float,
                 energy_max: float, energy_mean: float, energy_sigma: float,
                 n_pitch_bin: int, n_energy_bin: int, n_dec_layer: int,
                 n_dec_head: int, n_dec_conv_filter: int,
                 dec_conv_kernel_size: int, dec_dropout: float, mel_dim: int,
                 n_speaker: int, postnet_kernel_size: int,
                 postnet_hidden_dim: int, n_postnet_conv_layers: int,
                 postnet_dropout: float) -> None:
        """Initializing FastSpeech2.

        Args:
            enc_hidden_dim (int): _description_
            n_enc_layer (int): Number of layers in encoder.
            n_enc_head (int):  Number of attention heads in encoder.
            n_enc_conv_filter (int): Number of convolution filters in the first
            convolution layer of encoder FFT block.
            enc_conv_kernel_size (List[int]): Kernel sizes of convolution layers
            in encoder FFT block.
            enc_dropout (float): Dropout of encoder.
            n_vocab (int): Vocabulary size of input sequences.
            padding_idx (int): Index of padding elements in vocabulary.
            n_va_conv_filter (int): Number of convolution filters for each
            variance predictor in variance adapter.
            va_conv_kernel_size (int): Kernel size for all convolution layers
            in variance adapter.
            va_dropout (float): Dropout of variance adapter.
            pitch_min (float): Minimum of pitch in linear domain.
            pitch_max (float): Maximum of pitch in linear domain.
            pitch_mean (float): Mean of pitch in linear domain.
            pitch_sigma (float): Sigma of pitch in linear domain.
            energy_min (float): Minimum of energy in linear domain.
            energy_max (float): Maximum of energy in linear domain.
            energy_mean (float): Mean of energy in linear domain.
            energy_sigma (float): Sigma of energy in linear domain.
            n_pitch_bin (int): Number of bins for pitch quantization.
            n_energy_bin (int): Number of bins for energy quantization.
            n_dec_layer (int): Number of layers in mel decoder.
            n_dec_head (int): Number of attention heads in mel decoder.
            n_dec_conv_filter (int): Number of convolution filters in the
            first convolution layer of decoder FFT block.
            dec_conv_kernel_size (int): Kernel sizes of convolution layers in
            decoder FFT block.
            dec_dropout (float): Dropout of decoder.
            mel_dim (int): Dimension of mel.
            n_speaker (int): Number of speakers. If n_speaker > 1, initialize a
            multi-speaker FastSpeech2.
        """
        super().__init__()
        if n_speaker > 1:
            self.speaker_embedding = nn.Embedding(n_speaker, enc_hidden_dim)
        else:
            self.speaker_embedding = None
        self.encoder = encoder.FastSpeech2Encoder(
            enc_hidden_dim, n_enc_layer, n_enc_head, n_enc_conv_filter,
            enc_conv_kernel_size, enc_dropout, n_vocab, padding_idx)
        self.variance_adapter = variance_adapter.VarianceAdaptor(
            enc_hidden_dim, n_va_conv_filter, va_conv_kernel_size, va_dropout,
            pitch_min, pitch_max, pitch_mean, pitch_sigma, energy_min,
            energy_max, energy_mean, energy_sigma, n_pitch_bin, n_energy_bin)
        self.decoder = decoder.FastSpeech2Decoder(enc_hidden_dim, n_dec_layer,
                                                  n_dec_head,
                                                  n_dec_conv_filter,
                                                  dec_conv_kernel_size,
                                                  dec_dropout)
        self.linear = nn.Linear(enc_hidden_dim, mel_dim)
        self.postnet = postnet.Postnet(mel_dim, postnet_kernel_size,
                                       postnet_hidden_dim,
                                       n_postnet_conv_layers, postnet_dropout)

    def _get_speaker_embedding(self, enc_output: torch.Tensor,
                               speaker: torch.Tensor) -> torch.Tensor:
        """Getting speaker embedding according to speak ID.

        If this is a multi-speaker FastSpeech2 and speaker IDs are given,
        speaker embeddings will be added to enc_output.

        Args:
            enc_output (torch.Tensor): Output from encoder of FastSpeech2.
            speaker (torch.Tensor): Speaker ID.

        Raises:
            ValueError: Raised when speaker ID is given to a single-speaker
            FastSpeech2 or no speaker ID is given to a multi-speaker
            FastSpeech2.

        Returns:
            torch.Tensor: Encoder output with speaker embedding added.
        """
        if self.speaker_embedding is not None:
            if speaker is None:
                raise ValueError(
                    ("Speaker ID is not given when using multi-speaker "
                     "Fastspeech2."))
            else:
                enc_output += self.speaker_embedding(speaker).unsqueeze(1)
        else:
            if speaker is not None:
                raise ValueError(
                    "Speaker ID is given when using single-speaker Fastspeech2."
                )
        return enc_output

    def forward(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor,
        x_token_type: torch.IntTensor,
        duration_target: torch.Tensor,
        pitch_target: torch.Tensor,
        energy_target: torch.Tensor,
        speaker: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function of FastSpeech2 used for training.

        This function should not be used for inference. Teacher-forcing is
        applied to predict mel.

        Args:
            x (torch.Tensor): Input phoneme sequences.
            x_length (torch.Tensor): A Tensor of shape (b), each element is the
            length of x.
            x_token_type (torch.IntTensor): An IntTensor of shape (b,t'). Assume
            the output of the final FFT block is final_FFT_output and the final
            output of encoder is enc_output. When token_type[i,j] is 0,
            final_FFT_output[i,j,:] will be remove from enc_output.
            duration_target (torch.Tensor): shape (b,t). Ground truth duration
            for each phoneme.
            pitch_target (torch.Tensor): shape (b,t). Ground truth pitch for
            each phoneme.
            energy_target (torch.Tensor): shape (b,t). Ground truth energy for
            each phoneme.
            speaker (torch.Tensor, optional): shape (b). Speaker ID for each
            training sample. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            predicted mel, predicted pitch, predicted energy, predicted duration
            in log domain.
        """
        # If x_padding_mask[i,j] is True, x[i,j,:] will be masked in encoder
        # self-attention.
        x_padding_mask = utils.get_mask_from_lengths(x_length)

        enc_output, enc_output_seq_len, _ = self.encoder(
            x, x_padding_mask, x_token_type)
        enc_output_mask = utils.get_mask_from_lengths(enc_output_seq_len)
        enc_output = self._get_speaker_embedding(enc_output, speaker)
        (variance_adapter_output, mel_len, pitch_prediction, energy_prediction,
         log_duration_prediction) = self.variance_adapter(
             enc_output, enc_output_mask, duration_target, pitch_target,
             energy_target)
        mel_mask = utils.get_mask_from_lengths(mel_len)
        dec_output, _ = self.decoder(variance_adapter_output, mel_mask)
        mel_prediction = self.linear(dec_output)
        postnet_mel_prediction = self.postnet(mel_prediction) + mel_prediction
        return (mel_prediction, postnet_mel_prediction, mel_mask,
                pitch_prediction, energy_prediction, log_duration_prediction,
                enc_output_mask)

    def inference(self,
                  x: torch.Tensor,
                  x_length: torch.Tensor,
                  x_token_type: torch.IntTensor,
                  p_control: float,
                  e_control: float,
                  d_control: float,
                  speaker: torch.Tensor = None) -> torch.Tensor:
        """Inference function of FastSpeech2.

        Args:
            x (torch.Tensor): Input phoneme sequences.
            x_length (torch.Tensor): A Tensor of shape (b), each element is the
            length of x.
            x_token_type (torch.IntTensor): shape (b,t'). Assume the output of
            the final FFT
            block is final_FFT_output and the final output of encoder is
            enc_output. When token_type[i,j] is 0, final_FFT_output[i,j,:] will
            be remove from enc_output.
            p_control (float): shape (b,t). Ground truth duration for each
            phoneme.
            e_control (float): shape (b,t). Ground truth pitch for each phoneme.
            d_control (float): shape (b,t). Ground truth energy for each
            phoneme.
            speaker (torch.Tensor, optional): shape (b). Speaker ID for each
            training sample. Defaults to None.
        Returns:
            torch.Tensor: Predicted mel.
        """
        # If x_padding_mask[i,j] is True, x[i,j,:] will be masked in encoder
        # self-attention.
        x_padding_mask = utils.get_mask_from_lengths(x_length)

        enc_output, enc_output_seq_len, _ = self.encoder.inference(
            x, x_padding_mask, x_token_type)
        enc_output_mask = utils.get_mask_from_lengths(enc_output_seq_len)

        enc_output = self._get_speaker_embedding(enc_output, speaker)

        variance_adapter_output, mel_len = self.variance_adapter.inference(
            enc_output, enc_output_mask, p_control, e_control, d_control)
        # If mel_mask[i,j] is True, variance_adapter_output[i,j,:] will be
        # masked in decoder self-attention.

        mel_mask = utils.get_mask_from_lengths(mel_len)
        dec_output, _ = self.decoder.inference(variance_adapter_output,
                                               mel_mask)
        mel_prediction = self.linear(dec_output)
        mel_prediction = self.postnet(mel_prediction) + mel_prediction
        return mel_prediction
