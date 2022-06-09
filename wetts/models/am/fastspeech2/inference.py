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

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader
from yacs import config
import numpy as np

from wetts.models.am.fastspeech2.fastspeech2 import FastSpeech2
from wetts.models.am.fastspeech2.train import load_ckpt
from wetts.models.am.fastspeech2.module.dataset import (
    FastSpeech2InferenceDataset)


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description='FastSpeech2 inference script.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='num worker to read the data')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--text_file', required=True, help='test data list')
    parser.add_argument('--speaker_file',
                        required=True,
                        help='speaker for each utterance')
    parser.add_argument('--lexicon_file', required=True, help='lexicon file')
    parser.add_argument('--cmvn_dir',
                        required=True,
                        help='mel/energy/pitch cmvn dir')
    parser.add_argument('--spk2id_file',
                        required=True,
                        help='speaker to id file')
    parser.add_argument('--phn2id_file',
                        required=True,
                        help='phone to id file')
    parser.add_argument('--special_token_file',
                        required=True,
                        help='special tokens file')
    parser.add_argument('--p_control',
                        type=float,
                        default=1.0,
                        help='Pitch manipulation factor.')
    parser.add_argument('--e_control',
                        type=float,
                        default=1.0,
                        help='Energy manipulation factor.')
    parser.add_argument('--d_control',
                        type=float,
                        default=1.0,
                        help='Duration manipulation factor.')
    parser.add_argument('--export_dir')
    parser.add_argument('--ckpt')
    args = parser.parse_args(argv)
    return args


def main(args):
    with open(args.config, 'r') as fin:
        conf = config.load_cfg(fin)
    export_dir = pathlib.Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    (dataset, mel_stats, pitch_stats, energy_stats,
     phn2id) = FastSpeech2InferenceDataset(args.text_file, args.speaker_file,
                                           args.special_token_file,
                                           args.lexicon_file, args.phn2id_file,
                                           args.spk2id_file, args.cmvn_dir,
                                           args.batch_size)
    mel_mean, mel_sigma = mel_stats
    pitch_mean, pitch_sigma, pitch_min, pitch_max = pitch_stats
    energy_mean, energy_sigma, energy_min, energy_max = energy_stats

    data_loader = DataLoader(dataset,
                             batch_size=None,
                             num_workers=args.num_workers)
    model = FastSpeech2(
        conf.model.d_model, conf.model.n_enc_layer, conf.model.n_enc_head,
        conf.model.n_enc_conv_filter, conf.model.enc_conv_kernel_size,
        conf.model.enc_dropout, len(phn2id), conf.model.padding_idx,
        conf.model.n_va_conv_filter, conf.model.va_conv_kernel_size,
        conf.model.va_dropout, pitch_min, pitch_max, pitch_mean, pitch_sigma,
        energy_min, energy_max, energy_mean, energy_sigma,
        conf.model.n_pitch_bin, conf.model.n_energy_bin,
        conf.model.n_dec_layer, conf.model.n_dec_head,
        conf.model.n_dec_conv_filter, conf.model.dec_conv_kernel_size,
        conf.model.dec_dropout, conf.n_mels, conf.n_speaker,
        conf.model.postnet_kernel_size, conf.model.postnet_hidden_dim,
        conf.model.n_postnet_conv_layers, conf.model.postnet_dropout,
        conf.model.max_pos_enc_len)

    model_state_dict, _, _, _, _, epoch = load_ckpt(args.ckpt)
    print('loading FastSpeech2 ckpt, epoch {}'.format(epoch))
    model.load_state_dict(model_state_dict)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        output = []
        for i, x in enumerate(data_loader):
            text, text_length, token_types, speakers = x
            text = text.cuda()
            text_length = text_length.cuda()
            token_types = token_types.cuda()
            speakers = speakers.cuda()

            (_, postnet_mel_prediction, mel_mask,
             *_) = model(text,
                         text_length,
                         token_types,
                         p_control=args.p_control,
                         d_control=args.d_control,
                         e_control=args.e_control,
                         speaker=speakers)
            for mel, l in zip(postnet_mel_prediction, (~mel_mask).sum(dim=1)):
                output.append(mel[:l].cpu().numpy() * mel_sigma + mel_mean)
        for i, mel in enumerate(output):
            np.save(export_dir / '{}.npy'.format(i), mel)


if __name__ == '__main__':
    main(get_args())
