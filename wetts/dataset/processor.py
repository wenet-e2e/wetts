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
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torchaudio

from wetts.utils import constants


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.
        Args:
            data(Iterable[str]): url or local file list
        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix
        Args:
            data: Iterable[{src, stream}]
        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'json':
                        json_obj = json.load(file_obj)
                        for k, v in json_obj.items():
                            example[k] = v
                    elif postfix in constants.AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


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
