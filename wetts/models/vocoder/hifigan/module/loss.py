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
# Modified from HiFi-GAN(https://github.com/jik876/hifi-gan)

from librosa.filters import mel as librosa_mel_fn

import torch
from torch import nn


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

# TODO: Using pytorch to get mel-spectrogram for both feature extraction and
# loss calculation.

class HiFiGANMelLoss(nn.Module):

    def __init__(self,
                 sr,
                 n_fft,
                 num_mels,
                 hop_length,
                 win_length,
                 fmin,
                 fmax=None):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        # mel_basis: n_mel, n_fft+1
        self.mel_basis = nn.Parameter(torch.from_numpy(
            librosa_mel_fn(sr=sr,
                           n_fft=n_fft,
                           n_mels=num_mels,
                           fmin=fmin,
                           fmax=fmax)), requires_grad=False)
        self.hann_window = nn.Parameter(torch.hann_window(win_length),
                                        requires_grad=False)
        self.l1_loss = nn.L1Loss()

    def forward(self, y_prediction, y):
        return self.l1_loss(self._get_mel(y_prediction), self._get_mel(y))

    def _get_mel(self, x):
        x = nn.functional.pad(x.unsqueeze(1),
                              (int((self.n_fft - self.hop_length) / 2),
                               int((self.n_fft - self.hop_length) / 2)),
                              mode='reflect').squeeze(1)
        spec = torch.stft(x,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.hann_window,
                          center=False,
                          pad_mode='reflect',
                          return_complex=True,
                          normalized=False,
                          onesided=True)
        spec = torch.matmul(self.mel_basis, torch.abs(spec))
        spec = torch.log(torch.clamp(spec, min=1e-5))
        return spec
