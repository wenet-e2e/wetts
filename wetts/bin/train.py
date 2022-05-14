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

from torch.utils.data import DataLoader

from wetts.dataset.dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='num worker to read the data')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_list', required=True, help='train data list')
    parser.add_argument('--cmvn_dir',
                        required=True,
                        help='mel/energy/f0 cmvn dir')
    parser.add_argument('--spk2id_file',
                        required=True,
                        help='speaker to id file')
    parser.add_argument('--phn2id_file',
                        required=True,
                        help='phone to id file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    dataset = Dataset(args.train_list, args.spk2id_file, args.phn2id_file,
                      args.cmvn_dir, configs)

    data_loader = DataLoader(dataset,
                             batch_size=None,
                             num_workers=args.num_workers)

    for i, x in enumerate(data_loader):
        print(x)
        if i > 1:
            break


if __name__ == '__main__':
    main()
