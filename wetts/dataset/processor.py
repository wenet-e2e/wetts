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

import librosa
import numpy as np
import torchaudio
from yacs.config import CfgNode

from wetts.dataset.feats import Energy, LogMelFBank, Pitch

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


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
                    elif postfix in AUDIO_FORMAT_SETS:
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


def merge_silence(data):
    """ merge silences
        Args:
            data: Iterable[{key, wav, speaker, durations, phones}]
        Returns:
            Iterable[{key, wav, speaker, durations, phones}]
    """
    for sample in data:
        cur_phn, cur_dur = sample['phones'], sample['durations']
        new_phn = []
        new_dur = []

        # merge sp and sil
        for i, p in enumerate(cur_phn):
            if i > 0 and 'sil' == p and cur_phn[i - 1] in {"sil", "sp"}:
                print(sample)
                new_dur[-1] += cur_dur[i]
                new_phn[-1] = 'sil'
            else:
                new_phn.append(p)
                new_dur.append(cur_dur[i])

        assert len(new_phn) == len(new_dur)
        sample['durations'] = new_dur
        sample['phones'] = new_phn
        yield sample


def compute_feats(data, config):
    """ Compute mel, f0, energy feature
        Args:
            data: Iterable[{key, wav, speaker, durations, phones}]
        Returns:
            Iterable[{key, wav, speaker, durations, phones, mel, f0, energy}]
    """
    cut_sil = config.get('cut_sil', True)
    config = CfgNode(config)
    mel_extractor = LogMelFBank(sr=config.fs,
                                n_fft=config.n_fft,
                                hop_length=config.n_shift,
                                win_length=config.win_length,
                                window=config.window,
                                n_mels=config.n_mels,
                                fmin=config.fmin,
                                fmax=config.fmax)
    pitch_extractor = Pitch(sr=config.fs,
                            hop_length=config.n_shift,
                            f0min=config.f0min,
                            f0max=config.f0max)
    energy_extractor = Energy(sr=config.fs,
                              n_fft=config.n_fft,
                              hop_length=config.n_shift,
                              win_length=config.win_length,
                              window=config.window)
    for sample in data:
        key = sample['key']
        wav = sample['wav'].numpy()[0]  # First channel
        phones = sample['phones']
        durations = sample['durations']
        assert len(wav.shape) == 1, f'{key} is not a mono-channel audio.'
        assert np.abs(wav).max(
        ) <= 1.0, f"{key} is seems to be different that 16 bit PCM."

        d_cumsum = np.pad(np.array(durations).cumsum(0), (1, 0), 'constant')
        # little imprecise than use *.TextGrid directly
        times = librosa.frames_to_time(d_cumsum,
                                       sr=config.fs,
                                       hop_length=config.n_shift)
        if cut_sil:
            start = 0
            end = d_cumsum[-1]
            if phones[0] == "sil" and len(durations) > 1:
                start = times[1]
                durations = durations[1:]
                phones = phones[1:]
            if phones[-1] == 'sil' and len(durations) > 1:
                end = times[-2]
                durations = durations[:-1]
                phones = phones[:-1]
            start, end = librosa.time_to_samples([start, end], sr=config.fs)
            wav = wav[start:end]

        # extract mel feats
        logmel = mel_extractor.get_log_mel_fbank(wav)
        num_frames = logmel.shape[0]
        diff = num_frames - sum(durations)
        if diff != 0:
            if diff > 0:
                durations[-1] += diff
            elif durations[-1] + diff > 0:
                durations[-1] += diff
            elif durations[0] + diff > 0:
                durations[0] += diff
            else:
                print('Ignore utterance {}'.format(key))
                continue
        assert sum(durations) == num_frames
        # extract f0
        f0 = pitch_extractor.get_pitch(wav, duration=np.array(durations))
        assert f0.shape[0] == len(durations)
        # extract energy
        energy = energy_extractor.get_energy(wav, duration=np.array(durations))
        assert energy.shape[0] == len(durations)
        sample['durations'] = durations
        sample['mel'] = logmel
        sample['f0'] = f0
        sample['energy'] = energy
        yield sample


def shuffle(data, shuffle_size=1500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav, speaker, sample_rate}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav, speaker, sample_rate}]
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
