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

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ProsodyDataset, collote_fn, IGNORE_ID
from prosody_model import ProsodyModel


class ClassificationMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.stats = [None] * num_classes
        for i in range(num_classes):
            self.stats[i] = {}
            self.stats[i]['total_ref'] = 0
            self.stats[i]['total_hyp'] = 0
            self.stats[i]['correct'] = 0

    def add_stat(self, pred, label):
        assert len(pred) == len(label)
        for i in range(len(pred)):
            self.stats[pred[i]]['total_hyp'] += 1
            self.stats[label[i]]['total_ref'] += 1
            if pred[i] == label[i]:
                self.stats[pred[i]]['correct'] += 1

    def report(self):
        print('class\t\tprecision\t\trecall\t\tf1-score')
        for i in range(self.num_classes):
            tag = '#{}'.format(i)
            precision = self.stats[i]['correct'] / (
                self.stats[i]['total_hyp'] + 1e-6)
            recall = self.stats[i]['correct'] / (self.stats[i]['total_ref'] +
                                                 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            print('{}\t\t{:.6f}\t\t{:.6f}\t\t{:.6f}'.format(
                tag, precision, recall, f1))


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--num_prosody',
                        type=int,
                        default=4,
                        help='num prosody classes')
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
    test_data = ProsodyDataset(args.test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 collate_fn=collote_fn)
    # Init model
    model = ProsodyModel(args.num_prosody)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    metric = ClassificationMetric(args.num_prosody)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_dataloader))
        for batch, (inputs, labels) in enumerate(test_dataloader):
            logits = model(inputs)
            mask = labels != IGNORE_ID
            lengths = torch.sum(mask, dim=1)
            for i in range(logits.size(0)):
                # Remove [CLS], [SEP] and padding
                pred = logits[i][1:lengths[i] + 1, :].argmax(-1).tolist()
                label = labels[i][1:lengths[i] + 1].tolist()
                metric.add_stat(pred, label)
            pbar.update(1)
        pbar.close()

    metric.report()


if __name__ == '__main__':
    main()
