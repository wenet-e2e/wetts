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
from wetts.utils.file_utils import read_lists
from wetts.models.am.fastspeech2.module.dataset import compute_feats


def load_finetune_files(data):
    for sample in data:
        sample = sample['src']
        assert 'mel_prediction_filepath' in sample
        assert 'wav_target_filepath' in sample

        sample['mel'] = np.load(sample['mel_prediction_filepath'])
        # assume the range of wav is [-1,1] and is resampled to target sample
        # rate 22050
        sample['sample_rate'], sample['wav'] = scipy.io.wavfile.read(
            sample['wav_target_filepath'])

        sample['wav'] = torch.from_numpy(sample['wav'])
        yield sample


def get_mel_wav_clip(data,
                     segment_size=8192,
                     hop_length=256,
                     num_train_clips=0):
    for sample in data:
        sample['mel'] = torch.from_numpy(sample['mel']).float()
        mel_clip_len = segment_size // hop_length
        mel_len = sample['mel'].size(0)
        # idx_list will contain some segments which are shorter than segment_size
        # because we want hifigan to see samples with padding.
        idx_list = list(
            range(random.randrange(0, mel_clip_len), mel_len, mel_clip_len))
        random.shuffle(idx_list)
        for i, start in enumerate(idx_list):
            # if num_train_clips is 0, we will use all clips in a single wav
            # otherwise, we will only use num_train_clips clips in a single wav
            if num_train_clips != 0 and i > num_train_clips - 1:
                break
            sample['mel_clip'] = sample['mel'][start:start + mel_clip_len]
            sample['wav_clip'] = sample['wav'][start *
                                               hop_length:(start +
                                                           mel_clip_len) *
                                               hop_length]
            yield sample


def padding_clip(data, segment_size=8192, hop_length=256):
    mel_len = segment_size // hop_length
    for samples in data:
        padded_mel_clip = []
        padded_wav_clip = []
        mels = []
        wavs = []
        for sample in samples:
            # mels and wavs are padded here because we may select some mel
            # segments and wav segments shorter than segment_size
            if sample['mel_clip'].size(0) != mel_len:
                sample['mel_clip'] = F.pad(
                    sample['mel_clip'],
                    (0, 0, 0, mel_len - sample['mel_clip'].size(0)))

            if sample['wav_clip'].size(0) != segment_size:
                sample['wav_clip'] = F.pad(
                    sample['wav_clip'],
                    (0, segment_size - sample['wav_clip'].size(0)))
            padded_mel_clip.append(sample['mel_clip'])
            padded_wav_clip.append(sample['wav_clip'])
            # full mels are preserved because we want to use it in validation to
            # log one sample to tensorboard
            mels.append(sample['mel'])
            # full wavs are preserved to be used in validation
            wavs.append(sample['wav'])
        # padded_mel_clip: b,t,d
        padded_mel_clip = torch.stack(padded_mel_clip)
        # padded_wav_clip: b,t
        padded_wav_clip = torch.stack(padded_wav_clip)

        yield (padded_mel_clip, padded_wav_clip, mels, wavs)


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


def HiFiGANTrainingDataset(datalist_filepath, conf, batch_size=32):
    lists = read_lists(datalist_filepath)
    dataset = utils.DataList(lists, shuffle=conf.shuffle)
    dataset = utils.Processor(dataset, processor.parse_raw)
    dataset = utils.Processor(dataset, processor.resample, conf.sr)
    dataset = utils.Processor(dataset,
                              compute_feats,
                              conf,
                              extract_pitch=False,
                              extract_energy=False)
    dataset = utils.Processor(dataset,
                              get_mel_wav_clip,
                              segment_size=conf.segment_size,
                              hop_length=conf.hop_length,
                              num_train_clips=conf.num_train_clips)
    dataset = utils.Processor(dataset, processor.batch, batch_size)
    dataset = utils.Processor(dataset, padding_clip)
    return dataset


def HiFiGANFinetuneDataset(datalist_filepath, conf, batch_size=32):
    with jsonlines.open(datalist_filepath) as f:
        datalist = list(f)
    dataset = utils.DataList(datalist, shuffle=False)
    dataset = utils.Processor(dataset, load_finetune_files)
    dataset = utils.Processor(dataset,
                              get_mel_wav_clip,
                              segment_size=conf.segment_size,
                              hop_length=conf.hop_length,
                              num_train_clips=conf.num_train_clips)
    dataset = utils.Processor(dataset, processor.batch, batch_size=batch_size)
    dataset = utils.Processor(dataset,
                              padding_clip,
                              segment_size=conf.segment_size,
                              hop_length=conf.hop_length)
    return dataset


def HiFiGANInferenceDataset(datalist_filepath, batch_size=32):
    with jsonlines.open(datalist_filepath) as f:
        datalist = list(f)
    dataset = utils.DataList(datalist, shuffle=False)
    dataset = utils.Processor(dataset, load_inference_mel_files)
    dataset = utils.Processor(dataset, processor.batch, batch_size=batch_size)
    dataset = utils.Processor(dataset, padding_inference_mels)
    return dataset
