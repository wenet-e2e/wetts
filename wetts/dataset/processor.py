# Copyright (c) 2022 Horizon Robtics. (authors: Binbin Zhang)
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

import json
import random

import torchaudio


def apply_spk2id(data, spk2id):
    for sample in data:
        assert 'speaker' in sample
        sample['speaker'] = spk2id[sample['speaker']]
        yield sample


def apply_phn2id(data, phn2id):
    for sample in data:
        assert 'text' in sample
        sample['text'] = [phn2id[x] for x in sample['text']]
        yield sample


def shuffle(data, shuffle_size=1500):
    """ Local shuffle the data
        Args:
            data: Iterable[{}]
            shuffle_size: buffer size for shuffle
        Returns:
            Iterable[{}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def batch(data, batch_size=2):
    """ Static batch the data by `batch_size`
        Args:
            data: Iterable[{}]
            batch_size: batch size
        Returns:
            Iterable[List[{}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            # clamp here to force resampled audio in range [-1,1]
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=resample_rate)(waveform).clamp(min=-1, max=1)
        yield sample


def parse_raw(data):
    """Parsing raw json strings to features.

        Args:
            data: Iterable[str], str is a json line which contains one sample
        Returns:
            Iterable[{key, wav, text, speaker, duration}]
    """
    for sample in data:
        assert 'src' in sample
        obj = json.loads(sample['src'])
        assert 'key' in obj
        assert 'wav_path' in obj
        assert 'speaker' in obj
        assert 'text' in obj
        assert 'duration' in obj

        sample['key'] = obj['key']
        sample['wav'], sample['sample_rate'] = torchaudio.load(obj['wav_path'])
        sample['text'] = obj['text']
        sample['speaker'] = obj['speaker']
        sample['duration'] = [float(x) for x in obj['duration']]
        yield sample
