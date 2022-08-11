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

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

IGNORE_ID = -100

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


class ProsodyDataset(Dataset):
    def __init__(self, training_file):
        self.data = self.read_training_data(training_file)

    # The format of the training_file is like
    # 今天 #1 天气 #1 怎么样 #3
    def read_training_data(self, training_file):
        data = []
        with open(training_file, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                # Check
                ok = True
                if len(arr) % 2 != 0:
                    ok = False
                else:
                    for i in range(0, len(arr), 2):
                        if arr[i + 1][0] != '#' or \
                           not arr[i + 1][1:].encode('utf8').isdigit():
                            ok = False
                            break
                if not ok:
                    print('Ignore line {}'.format(line.strip()))
                    continue

                tokens = []
                prosody = []
                for i in range(0, len(arr), 2):
                    toks = tokenizer.encode(arr[i], add_special_tokens=False)
                    rank = int(arr[i + 1][1:])
                    rhythm = [0] * len(toks)
                    rhythm[-1] = rank
                    prosody.extend(rhythm)
                    tokens.append(arr[i])
                data.append({'sentence': tokens, 'prosody': prosody})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    batch_sentence, batch_prosody = [], []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_prosody.append(sample['prosody'])
    batch_inputs = tokenizer(batch_sentence,
                             padding=True,
                             truncation=True,
                             is_split_into_words=True,
                             return_tensors="pt")
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for idx, prosody in enumerate(batch_prosody):
        batch_label[idx][0] = IGNORE_ID  # [CLS]
        batch_label[idx][len(prosody):] = IGNORE_ID  # [SEP] add padding
        batch_label[idx][1:len(prosody) + 1] = np.array(prosody,
                                                        dtype=np.int32)
    return batch_inputs, torch.tensor(batch_label)


if __name__ == '__main__':
    import sys
    train_data = ProsodyDataset(sys.argv[1])
    train_dataloader = DataLoader(train_data,
                                  batch_size=2,
                                  shuffle=False,
                                  collate_fn=collote_fn)
    for x in train_dataloader:
        print(x)
