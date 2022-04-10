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
import yaml

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from wetts.dataset.dataset import CmvnDataset


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
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    dataset = CmvnDataset(args.input_list, configs)
    mel_scaler = StandardScaler()
    f0_scaler = StandardScaler()
    energy_scaler = StandardScaler()

    data_loader = DataLoader(dataset,
                             batch_size=None,
                             num_workers=args.num_workers)

    with tqdm(total=args.total) as progress:
        for i, x in enumerate(data_loader):
            mel_scaler.partial_fit(x['mel'])
            f0_scaler.partial_fit(x['f0'])
            energy_scaler.partial_fit(x['energy'])
            progress.update()

    mel_stats = np.stack([mel_scaler.mean_, mel_scaler.scale_], axis=0)
    np.savetxt(os.path.join(args.output_dir, 'mel_cmvn.txt'),
               mel_stats.astype(np.float32))
    f0_stats = np.stack([f0_scaler.mean_, f0_scaler.scale_], axis=0)
    np.savetxt(os.path.join(args.output_dir, 'f0_cmvn.txt'),
               f0_stats.astype(np.float32))
    energy_stats = np.stack([energy_scaler.mean_, energy_scaler.scale_],
                            axis=0)
    np.savetxt(os.path.join(args.output_dir, 'energy_cmvn.txt'),
               energy_stats.astype(np.float32))


if __name__ == '__main__':
    main()
