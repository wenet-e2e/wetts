# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com)
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
import os
from yacs import config

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from wetts.models.am.fastspeech2.module.dataset import CmvnDataset


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--total', type=int, default=1e5, help='total record')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='num worker to read the data')
    parser.add_argument('config', help='config file')
    parser.add_argument('input_list', help='input data list')
    parser.add_argument('output_dir', help='output dir')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.config, 'r') as fin:
        conf = config.load_cfg(fin)
    dataset = CmvnDataset(args.input_list, conf)
    mel_scaler = StandardScaler()
    pitch_scaler = StandardScaler()
    pitch_minmax = MinMaxScaler()
    energy_scaler = StandardScaler()
    energy_minmax = MinMaxScaler()

    data_loader = DataLoader(dataset,
                             batch_size=None,
                             num_workers=args.num_workers)

    with tqdm(total=args.total) as progress:
        for i, x in enumerate(data_loader):
            mel, pitch, energy = x['mel'], x['pitch'].reshape(
                -1, 1), x['energy'].reshape(-1, 1)
            mel_scaler.partial_fit(mel)
            pitch_scaler.partial_fit(pitch)
            energy_scaler.partial_fit(energy)
            pitch_minmax.partial_fit(pitch)
            energy_minmax.partial_fit(energy)
            progress.update()

    mel_stats = np.stack([mel_scaler.mean_, mel_scaler.scale_], axis=0)
    np.savetxt(os.path.join(args.output_dir, 'mel_cmvn.txt'),
               mel_stats.astype(np.float32))
    pitch_stats = np.stack([
        pitch_scaler.mean_, pitch_scaler.scale_, pitch_minmax.data_min_,
        pitch_minmax.data_max_
    ], axis=0)
    np.savetxt(os.path.join(args.output_dir, 'pitch_cmvn.txt'),
               pitch_stats.astype(np.float32))
    energy_stats = np.stack([
        energy_scaler.mean_, energy_scaler.scale_, energy_minmax.data_min_,
        energy_minmax.data_max_
    ], axis=0)
    np.savetxt(os.path.join(args.output_dir, 'energy_cmvn.txt'),
               energy_stats.astype(np.float32))


if __name__ == '__main__':
    main()
