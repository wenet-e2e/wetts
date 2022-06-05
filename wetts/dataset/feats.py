# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#               2022 Tsinghua University (Jie Chen)
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
# Modified from espnet(https://github.com/espnet/espnet)
import librosa
import numpy as np
import pyworld
from scipy.interpolate import interp1d


class LogMelFBank():

    def __init__(self,
                 sr=24000,
                 n_fft=2048,
                 hop_length=300,
                 win_length=None,
                 window="hann",
                 n_mels=80,
                 fmin=80,
                 fmax=7600):
        self.sr = sr
        # stft
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = True
        self.pad_mode = "reflect"

        # mel
        self.n_mels = n_mels
        self.fmin = 0 if fmin is None else fmin
        self.fmax = sr / 2 if fmax is None else fmax

        self.mel_filter = self._create_mel_filter()

    def _create_mel_filter(self):
        mel_filter = librosa.filters.mel(sr=self.sr,
                                         n_fft=self.n_fft,
                                         n_mels=self.n_mels,
                                         fmin=self.fmin,
                                         fmax=self.fmax)
        return mel_filter

    def _stft(self, wav):
        D = librosa.core.stft(wav,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              window=self.window,
                              center=self.center,
                              pad_mode=self.pad_mode)
        return D

    def _spectrogram(self, wav):
        D = self._stft(wav)
        return np.abs(D)

    def _mel_spectrogram(self, wav):
        S = self._spectrogram(wav)
        mel = np.dot(self.mel_filter, S)
        return mel

    def get_log_mel_fbank(self, wav):
        mel = self._mel_spectrogram(wav)
        mel = np.clip(mel, a_min=1e-10, a_max=float("inf"))
        mel = np.log(mel.T)
        # (num_frames, n_mels)
        return mel


class Pitch():

    def __init__(self, sr=24000, hop_length=300, pitch_min=80, pitch_max=7600):

        self.sr = sr
        self.hop_length = hop_length
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

    def _convert_to_continuous_pitch(self, pitch: np.array) -> np.array:
        if (pitch == 0).all():
            print("All frames seems to be unvoiced.")
            return pitch

        # padding start and end of pitch sequence
        start_pitch = pitch[pitch != 0][0]
        end_pitch = pitch[pitch != 0][-1]
        start_idx = np.where(pitch == start_pitch)[0][0]
        end_idx = np.where(pitch == end_pitch)[0][-1]
        pitch[:start_idx] = start_pitch
        pitch[end_idx:] = end_pitch

        # get non-zero frame index
        nonzero_idxs = np.where(pitch != 0)[0]

        # perform linear interpolation
        interp_fn = interp1d(nonzero_idxs, pitch[nonzero_idxs])
        pitch = interp_fn(np.arange(0, pitch.shape[0]))

        return pitch

    def _calculate_pitch(self,
                         input: np.array,
                         use_continuous_pitch=True,
                         use_log_pitch=False) -> np.array:
        input = input.astype(np.float)
        frame_period = 1000 * self.hop_length / self.sr

        pitch, timeaxis = pyworld.dio(input,
                                      fs=self.sr,
                                      f0_floor=self.pitch_min,
                                      f0_ceil=self.pitch_max,
                                      frame_period=frame_period)
        pitch = pyworld.stonemask(input, pitch, timeaxis, self.sr)
        if use_continuous_pitch:
            pitch = self._convert_to_continuous_pitch(pitch)
        if use_log_pitch:
            nonzero_idxs = np.where(pitch != 0)[0]
            pitch[nonzero_idxs] = np.log(pitch[nonzero_idxs])
        return pitch.reshape(-1)

    def _average_by_duration(self, input: np.array, d: np.array) -> np.array:
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            mask = arr == 0
            arr[mask] = 0
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)

        # shape : (T)
        arr_list = np.array(arr_list)

        return arr_list

    def get_pitch(self,
                  wav,
                  use_continuous_pitch=True,
                  use_log_pitch=False,
                  use_token_averaged_pitch=True,
                  duration=None):
        pitch = self._calculate_pitch(wav, use_continuous_pitch, use_log_pitch)
        if use_token_averaged_pitch and duration is not None:
            pitch = self._average_by_duration(pitch, duration)
        return pitch


class Energy():

    def __init__(self,
                 sr=24000,
                 n_fft=2048,
                 hop_length=300,
                 win_length=None,
                 window="hann",
                 center=True,
                 pad_mode="reflect"):

        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def _stft(self, wav):
        D = librosa.core.stft(wav,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              window=self.window,
                              center=self.center,
                              pad_mode=self.pad_mode)
        return D

    def _calculate_energy(self, input):
        input = input.astype(np.float32)
        input_stft = self._stft(input)
        input_power = np.abs(input_stft)**2
        energy = np.sqrt(
            np.clip(np.sum(input_power, axis=0),
                    a_min=1.0e-10,
                    a_max=float('inf')))
        return energy

    def _average_by_duration(self, input: np.array, d: np.array) -> np.array:
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)
        # shape (T)
        arr_list = np.array(arr_list)
        return arr_list

    def get_energy(self, wav, use_token_averaged_energy=True, duration=None):
        energy = self._calculate_energy(wav)
        if use_token_averaged_energy and duration is not None:
            energy = self._average_by_duration(energy, duration)
        return energy
