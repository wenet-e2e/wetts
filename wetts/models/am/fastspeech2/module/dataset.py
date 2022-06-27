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

import pathlib
from typing import Set, Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import librosa

from wetts.utils import constants
from wetts.dataset import utils, processor, feats
from wetts.utils.file_utils import read_key2id, read_lists, read_lexicon


def padding_training_samples(data):
    """ Padding the data
        Args:
            data: Iterable[List[{key, wav, speaker, duration, text, mel,
                            pitch, energy, token_types, wav}]]
        Returns:
            Iterable[Tuple(keys, speaker, duration, text, mel, pitch, energy,
                           text_length, mel_length, token_types, sorted_wav)]
    """
    for sample in data:
        assert isinstance(sample, list)
        text_length = torch.tensor([len(x['text']) for x in sample],
                                   dtype=torch.int32)
        order = torch.argsort(text_length, descending=True)
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_speaker = torch.tensor([sample[i]['speaker'] for i in order],
                                      dtype=torch.int32)
        sorted_duration = [
            torch.tensor(sample[i]['duration'], dtype=torch.int32)
            for i in order
        ]
        sorted_text = [
            torch.tensor(sample[i]['text'], dtype=torch.int32) for i in order
        ]
        sorted_mel = [torch.from_numpy(sample[i]['mel']) for i in order]
        sorted_pitch = [torch.from_numpy(sample[i]['pitch']) for i in order]
        sorted_energy = [torch.from_numpy(sample[i]['energy']) for i in order]
        sorted_text_length = torch.tensor(
            [len(sample[i]['text']) for i in order], dtype=torch.int32)
        sorted_mel_length = torch.tensor(
            [sample[i]['mel'].shape[0] for i in order], dtype=torch.int32)

        sorted_token_types = [
            torch.tensor(sample[i]['token_types'], dtype=torch.int32)
            for i in order
        ]
        sorted_wav = [sample[i]['wav'] for i in order]

        padded_duration = pad_sequence(sorted_duration,
                                       batch_first=True,
                                       padding_value=0)
        padded_text = pad_sequence(sorted_text,
                                   batch_first=True,
                                   padding_value=0)
        padded_mel = pad_sequence(sorted_mel,
                                  batch_first=True,
                                  padding_value=0.0)
        padded_pitch = pad_sequence(sorted_pitch,
                                    batch_first=True,
                                    padding_value=0.0)
        padded_energy = pad_sequence(sorted_energy,
                                     batch_first=True,
                                     padding_value=0.0)
        padded_token_types = pad_sequence(sorted_token_types,
                                          batch_first=True,
                                          padding_value=0)

        yield (sorted_keys, sorted_speaker, padded_duration, padded_text,
               padded_mel, padded_pitch, padded_energy, sorted_text_length,
               sorted_mel_length, padded_token_types, sorted_wav)


def padding_inference_samples(data):
    for sample in data:
        assert isinstance(sample, list)
        text_length = torch.tensor([len(x['text']) for x in sample],
                                   dtype=torch.int32)
        order = torch.argsort(text_length, descending=True)
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_speaker = torch.tensor([sample[i]['speaker'] for i in order],
                                      dtype=torch.int32)
        sorted_text = [
            torch.tensor(sample[i]['text'], dtype=torch.int32) for i in order
        ]
        sorted_text_length = torch.tensor(
            [len(sample[i]['text']) for i in order], dtype=torch.int32)
        sorted_token_types = [
            torch.tensor(sample[i]['token_types'], dtype=torch.int32)
            for i in order
        ]
        padded_text = pad_sequence(sorted_text,
                                   batch_first=True,
                                   padding_value=0)
        padded_token_types = pad_sequence(sorted_token_types,
                                          batch_first=True,
                                          padding_value=0)

        yield (sorted_keys, padded_text, sorted_text_length,
               padded_token_types, sorted_speaker)


def apply_cmvn(data, mel_stats, pitch_stats, energy_stats):
    """ Apply CMVN on data
    """
    for sample in data:
        assert 'mel' in sample
        assert 'pitch' in sample
        assert 'energy' in sample
        sample['mel'] = (sample['mel'] - mel_stats[0]) / mel_stats[1]
        sample['pitch'] = (sample['pitch'] - pitch_stats[0]) / pitch_stats[1]
        sample['energy'] = (sample['energy'] -
                            energy_stats[0]) / energy_stats[1]
        yield sample


def generate_token_types(data, special_tokens: Set) -> Iterable:
    """Generating a sequence of binary values to indicate sepcial tokens in
    a phoneme sequence. 1 indicates a normal phoneme, and 0 indicates a special
    token. This function should be applied before apply_phn2id.

    Args:
        data: A batch of samples.
        special_tokens (Set): A set containing special tokens.

    Yields:
        Iterable[{key, wav, speaker, duration, text, mel, pitch, energy}]
    """
    for sample in data:
        assert 'text' in sample
        sample['token_types'] = [
            0 if x in special_tokens else 1 for x in sample['text']
        ]
        yield sample


def compute_feats(data, config):
    """ Compute mel, pitch, energy feature
        Args:
            data: Iterable[{key, wav, speaker, duration, text}]
        Returns:
            Iterable[{key, wav, speaker, duration, text, mel, pitch, energy}]
    """
    mel_extractor = feats.LogMelFBank(sr=config.sr,
                                      n_fft=config.n_fft,
                                      hop_length=config.hop_length,
                                      win_length=config.win_length,
                                      window=config.window,
                                      n_mels=config.n_mels,
                                      fmin=config.fmin,
                                      fmax=config.fmax)
    pitch_extractor = feats.Pitch(sr=config.sr,
                                  hop_length=config.hop_length,
                                  pitch_min=config.pitch_min,
                                  pitch_max=config.pitch_max)
    energy_extractor = feats.Energy(sr=config.sr,
                                    n_fft=config.n_fft,
                                    hop_length=config.hop_length,
                                    win_length=config.win_length,
                                    window=config.window)
    for sample in data:
        key = sample['key']
        wav = sample['wav'].numpy()[0]  # First channel
        text = sample['text']
        duration = np.array(sample['duration'])
        assert len(wav.shape) == 1, f'{key} is not a mono-channel audio.'
        assert np.abs(wav).max(
        ) <= 1.0, f"{key} is seems to be different that 16 bit PCM."

        d_cumsum = duration.cumsum()

        if config.cut_sil:
            start = 0
            end = d_cumsum[-1]
            if text[0] == "sil" and len(duration) > 1:
                start = d_cumsum[1]
                duration = duration[1:]
                text = text[1:]
            if text[-1] == 'sil' and len(duration) > 1:
                end = d_cumsum[-2]
                duration = duration[:-1]
                text = text[:-1]
            start, end = librosa.time_to_samples([start, end], sr=config.sr)
            wav = wav[start:end]
        # clip here to avoid 0 duration
        duration = librosa.time_to_frames(
            duration, sr=config.sr, hop_length=config.hop_length).clip(min=1)

        sample['wav'] = torch.from_numpy(wav)
        # extract mel feats
        logmel = mel_extractor.get_log_mel_fbank(wav)
        num_frames = logmel.shape[0]
        diff = num_frames - sum(duration)
        if diff != 0:
            if diff > 0:
                duration[-1] += diff
            elif duration[-1] + diff > 0:
                duration[-1] += diff
            elif duration[0] + diff > 0:
                duration[0] += diff
            else:
                print('Ignore utterance {}'.format(key))
                continue
        assert sum(duration) == num_frames
        # extract pitch
        pitch = pitch_extractor.get_pitch(wav, duration=duration)
        assert pitch.shape[0] == len(duration)
        # extract energy
        energy = energy_extractor.get_energy(wav, duration=duration)
        assert energy.shape[0] == len(duration)
        sample['duration'] = duration
        sample['mel'] = logmel
        sample['pitch'] = pitch
        sample['energy'] = energy
        sample['text'] = text
        yield sample


def merge_silence(data):
    """ merge silences
        Args:
            data: Iterable[{key, wav, speaker, duration, text}]
        Returns:
            Iterable[{key, wav, speaker, duration, text}]
    """
    for sample in data:
        cur_phn, cur_dur = sample['text'], sample['duration']
        new_text = []
        new_dur = []

        # merge sp and sil
        for i, p in enumerate(cur_phn):
            if i > 0 and 'sil' == p and cur_phn[i -
                                                1] in constants.SILENCE_PHONES:
                new_dur[-1] += cur_dur[i]
                new_text[-1] = 'sil'
            else:
                new_text.append(p)
                new_dur.append(cur_dur[i])

        assert len(new_text) == len(new_dur)
        sample['duration'] = new_dur
        sample['text'] = new_text
        yield sample


def apply_lexicon(data, lexicon, special_tokens):
    for sample in data:
        assert 'src' in sample
        sample = sample['src']
        new_text = []
        for token in sample['text']:
            if token in lexicon:
                new_text.extend(lexicon[token])
            elif token in special_tokens:
                new_text.append(token)
            else:
                raise ValueError('Token {} not in lexicon or special tokens!')
        sample['text'] = new_text
        yield sample


def CmvnDataset(data_list_file, conf):
    lists = read_lists(data_list_file)
    dataset = utils.DataList(lists, shuffle=False)
    dataset = utils.Processor(dataset, processor.parse_raw)
    dataset = utils.Processor(dataset, processor.resample, conf.sr)
    dataset = utils.Processor(dataset, compute_feats, conf)
    return dataset


def FastSpeech2TrainingDataset(data_list_file, batch_size, spk2id_file,
                               phn2id_file, special_tokens_file, cmvn_dir,
                               conf):
    cmvn_dir = pathlib.Path(cmvn_dir)
    lists = read_lists(data_list_file)
    spk2id = read_key2id(spk2id_file)
    phn2id = read_key2id(phn2id_file)
    special_tokens = set(read_lists(special_tokens_file))

    dataset = utils.DataList(lists, shuffle=conf.shuffle)
    dataset = utils.Processor(dataset, processor.parse_raw)
    dataset = utils.Processor(dataset, processor.resample, conf.sr)
    dataset = utils.Processor(dataset, processor.shuffle, conf.shuffle)
    dataset = utils.Processor(dataset, compute_feats, conf)
    dataset = utils.Processor(dataset, generate_token_types, special_tokens)
    dataset = utils.Processor(dataset, processor.apply_spk2id, spk2id)
    dataset = utils.Processor(dataset, processor.apply_phn2id, phn2id)

    mel_stats = np.loadtxt(cmvn_dir / 'mel_cmvn.txt')
    pitch_stats = np.loadtxt(cmvn_dir / 'pitch_cmvn.txt')
    energy_stats = np.loadtxt(cmvn_dir / 'energy_cmvn.txt')
    dataset = utils.Processor(dataset, apply_cmvn, mel_stats, pitch_stats,
                              energy_stats)
    dataset = utils.Processor(dataset, processor.batch, batch_size)
    dataset = utils.Processor(dataset, padding_training_samples)
    return dataset, mel_stats, pitch_stats, energy_stats, phn2id, spk2id


def FastSpeech2InferenceDataset(text_file, speaker_file, special_token_file,
                                lexicon_file, phn2id_file, spk2id_file,
                                cmvn_dir, batch_size):
    cmvn_dir = pathlib.Path(cmvn_dir)
    text = read_lists(text_file)
    speaker = read_lists(speaker_file)
    spk2id = read_key2id(spk2id_file)
    phn2id = read_key2id(phn2id_file)
    special_tokens = set(read_lists(special_token_file))
    lexicon = read_lexicon(lexicon_file)
    data_list = [{
        'text': t.split(),
        'speaker': s,
        'key': i
    } for i, (t, s) in enumerate(zip(text, speaker))]

    dataset = utils.DataList(data_list, shuffle=False)
    dataset = utils.Processor(dataset,
                              apply_lexicon,
                              lexicon=lexicon,
                              special_tokens=special_tokens)
    dataset = utils.Processor(dataset, generate_token_types, special_tokens)
    dataset = utils.Processor(dataset, processor.apply_spk2id, spk2id)
    dataset = utils.Processor(dataset, processor.apply_phn2id, phn2id)

    mel_stats = np.loadtxt(cmvn_dir / 'mel_cmvn.txt')
    pitch_stats = np.loadtxt(cmvn_dir / 'pitch_cmvn.txt')
    energy_stats = np.loadtxt(cmvn_dir / 'energy_cmvn.txt')

    dataset = utils.Processor(dataset, processor.batch, batch_size)
    dataset = utils.Processor(dataset, padding_inference_samples)
    return dataset, mel_stats, pitch_stats, energy_stats, phn2id, spk2id
