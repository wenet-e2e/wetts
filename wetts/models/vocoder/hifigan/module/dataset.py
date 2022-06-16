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

import random
import os

import jsonlines
import numpy as np
import scipy.io.wavfile
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from wetts.dataset import utils, processor


def load_finetune_files(data):
    for sample in data:
        sample = sample['src']
        assert 'mel_prediction_filepath' in sample
        assert 'wav_target_filepath' in sample

        sample['mel_prediction'] = torch.from_numpy(
            np.load(sample['mel_prediction_filepath'])).float()
        # assume the range of wav is [-1,1] and is resampled to target sample
        # rate 22050
        sr, wav = scipy.io.wavfile.read(sample['wav_target_filepath'])
        sample['sample_rate'], sample['wav'] = sr, torch.from_numpy(
            wav).float()
        yield sample


def get_finetune_clip(data, clip_size=8192, hop_length=256, cache_size=4):
    for sample in data:
        for _ in range(cache_size):
            mel_clip_len = clip_size // hop_length
            mel_len = sample['mel_prediction'].size(0)
            start = random.randint(0, max(mel_len - mel_clip_len, 1))
            sample['mel_prediction_clip'] = sample['mel_prediction'][
                start:start + mel_clip_len]
            sample['wav_clip'] = sample['wav'][start *
                                               hop_length:(start +
                                                           mel_clip_len) *
                                               hop_length]
            yield sample


def padding_finetune_clip(data, clip_size=8192, hop_length=256):
    mel_len = clip_size // hop_length
    for samples in data:
        padded_mel_prediction_clip = []
        padded_wav_clip = []
        for sample in samples:
            if sample['mel_prediction_clip'].size(0) != mel_len:
                sample['mel_prediction_clip'] = F.pad(
                    sample['mel_prediction_clip'],
                    (0, 0, 0, mel_len - sample['mel_prediction_clip'].size(0)))

            if sample['wav_clip'].size(0) != clip_size:
                sample['wav_clip'] = F.pad(
                    sample['wav_clip'],
                    (0, clip_size - sample['wav_clip'].size(0)))
            padded_mel_prediction_clip.append(sample['mel_prediction_clip'])
            padded_wav_clip.append(sample['wav_clip'])
        # padded_mel_prediction_clip: b,t,d
        padded_mel_prediction_clip = torch.stack(padded_mel_prediction_clip)
        # padded_wav_clip: b,t
        padded_wav_clip = torch.stack(padded_wav_clip)
        yield (padded_mel_prediction_clip, padded_wav_clip)


def load_inference_mel_files(data):
    for sample in data:
        sample = sample['src']
        assert 'mel_prediction_filepath' in sample
        sample['name'] = os.path.basename(sample['mel_prediction_filepath'])
        sample['mel_prediction'] = torch.from_numpy(
            np.load(sample['mel_prediction_filepath'])).float()
        yield sample


def padding_inference_mels(data):
    for samples in data:
        mel_length = torch.tensor(
            [len(sample['mel_prediction']) for sample in samples])
        order = torch.argsort(mel_length, descending=True)

        sorted_names = [samples[i]['name'] for i in order]
        sorted_mel_prediction = pad_sequence(
            [samples[i]['mel_prediction'] for i in order], batch_first=True)
        sorted_mel_length = torch.tensor(
            [len(samples[i]['mel_prediction']) for i in order])
        yield sorted_names, sorted_mel_prediction, sorted_mel_length


def HiFiGANFinetuneDataset(datalist_filepath,
                           clip_size=8192,
                           hop_length=256,
                           cache_size=4,
                           batch_size=32):
    with jsonlines.open(datalist_filepath) as f:
        datalist = list(f)
    dataset = utils.DataList(datalist, shuffle=False)
    dataset = utils.Processor(dataset, load_finetune_files)
    dataset = utils.Processor(dataset,
                              get_finetune_clip,
                              clip_size=clip_size,
                              hop_length=hop_length,
                              cache_size=cache_size)
    dataset = utils.Processor(dataset, processor.batch, batch_size=batch_size)
    dataset = utils.Processor(dataset,
                              padding_finetune_clip,
                              clip_size=clip_size,
                              hop_length=hop_length)
    return dataset


def HiFiGANInferenceDataset(datalist_filepath, batch_size=32):
    with jsonlines.open(datalist_filepath) as f:
        datalist = list(f)
    dataset = utils.DataList(datalist, shuffle=False)
    dataset = utils.Processor(dataset, load_inference_mel_files)
    dataset = utils.Processor(dataset, processor.batch, batch_size=batch_size)
    dataset = utils.Processor(dataset, padding_inference_mels)
    return dataset
