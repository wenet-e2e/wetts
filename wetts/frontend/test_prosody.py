# Copyright (c) 2022, Binbin Zhang (binbzha@qq.com)
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

from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FrontendDataset, collote_fn, IGNORE_ID
from model import FrontendModel
from utils import read_table


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--phone_dict', required=True, help='phone dict file')
    parser.add_argument('--prosody_dict',
                        required=True,
                        help='train data file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    phone_dict = read_table(args.phone_dict)
    prosody_dict = read_table(args.prosody_dict)
    num_phones = len(phone_dict)
    num_prosody = len(prosody_dict)

    test_data = FrontendDataset(prosody_file=args.test_data,
                                prosody_dict=phone_dict)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 collate_fn=collote_fn)
    # Init model
    model = FrontendModel(num_phones, num_prosody)

    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_dataloader))
        pw_f1_score = []
        pph_f1_score = []
        iph_f1_score = []
        for batch, (inputs, _, labels) in enumerate(test_dataloader):
            _, logits = model(inputs)
            mask = labels != IGNORE_ID
            lengths = torch.sum(mask, dim=1)
            for i in range(logits.size(0)):
                # Remove [CLS], [SEP] and padding
                pred = logits[i][1:lengths[i] + 1, :].argmax(-1).tolist()
                label = labels[i][1:lengths[i] + 1].tolist()
                pw_f1_score.append(f1_score([1 if x > 0 else 0 for x in label],
                                            [1 if x > 0 else 0 for x in pred]))
                pph_f1_score.append(f1_score([1 if x > 1 else 0 for x in label],
                                             [1 if x > 1 else 0 for x in pred]))
                iph_f1_score.append(f1_score([1 if x > 2 else 0 for x in label],
                                             [1 if x > 2 else 0 for x in pred]))
            pbar.update(1)
        print("pw f1_score {} pph f1_score {} iph f1_score {}".format(
            sum(pw_f1_score) / len(pw_f1_score), sum(pph_f1_score) / len(pph_f1_score),
            sum(iph_f1_score) / len(iph_f1_score)))
        pbar.close()


if __name__ == '__main__':
    main()
