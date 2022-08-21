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


class FrontendDataset(Dataset):

    def __init__(self,
                 polyphone_file=None,
                 phone_dict=None,
                 prosody_file=None,
                 prosody_dict=None):
        self.data = []
        if polyphone_file is not None:
            assert phone_dict is not None
            data = self.read_polyphone_data(polyphone_file, phone_dict)
            print('Reading {} polyphone data'.format(len(data)))
            self.data += data

        if prosody_file is not None:
            assert prosody_dict is not None
            data = self.read_prosody_data(prosody_file, prosody_dict)
            print('Reading {} prosody data'.format(len(data)))
            self.data += data

    def read_polyphone_data(self, polyphone_file, phone_dict):
        """ If there is a polyphone, it is surrounded by ▁, such as
            ```
            宋代出现了▁le5▁燕乐音阶的记载
            2011年9月17日，爆发了▁le5▁占领华尔街示威活动
            ```
        """
        data = []
        with open(polyphone_file, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split('▁')
                # Check
                tokens = []
                all_phones = []
                for i in range(0, len(arr), 2):
                    toks = tokenizer.encode(arr[i], add_special_tokens=False)
                    phones = [IGNORE_ID] * len(toks)
                    if i + 1 < len(arr):
                        if arr[i + 1] in phone_dict:
                            phones[-1] = phone_dict[arr[i + 1]]
                        else:
                            print('Unknown phone {}'.format(arr[i + 1]))
                    all_phones.extend(phones)
                    tokens.append(arr[i])
                data.append({
                    'sentence': tokens,
                    'phones': all_phones,
                    'prosody': [IGNORE_ID] * len(all_phones),
                })
        return data

    def read_prosody_data(self, prosody_file, prosody_dict):
        """ Each line of prosody is like:
            ```
            今天 #1 天气 #1 怎么样 #3
            ```
        """
        data = []
        with open(prosody_file, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                # Check
                ok = True
                if len(arr) % 2 != 0:
                    ok = False
                else:
                    for i in range(0, len(arr), 2):
                        # #N and N is less than `prosody_dict`
                        if arr[i + 1][0] != '#' or \
                           not arr[i + 1][1:].encode('utf8').isdigit() or \
                           int(arr[i + 1][1:]) >= len(prosody_dict):
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
                data.append({
                    'sentence': tokens,
                    'phones': [IGNORE_ID] * len(prosody),
                    'prosody': prosody,
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    batch_sentence, batch_phones, batch_prosody = [], [], []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_phones.append(sample['phones'])
        batch_prosody.append(sample['prosody'])
    batch_inputs = tokenizer(batch_sentence,
                             padding=True,
                             truncation=True,
                             is_split_into_words=True,
                             return_tensors="pt")
    phone_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for idx, phones in enumerate(batch_phones):
        phone_label[idx][0] = IGNORE_ID  # [CLS]
        phone_label[idx][len(phones):] = IGNORE_ID  # [SEP] add padding
        phone_label[idx][1:len(phones) + 1] = np.array(phones, dtype=np.int32)
    prosody_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for idx, prosody in enumerate(batch_prosody):
        prosody_label[idx][0] = IGNORE_ID  # [CLS]
        prosody_label[idx][len(prosody):] = IGNORE_ID  # [SEP] add padding
        prosody_label[idx][1:len(prosody) + 1] = np.array(prosody,
                                                          dtype=np.int32)

    return batch_inputs, torch.tensor(phone_label), torch.tensor(prosody_label)


if __name__ == '__main__':
    import sys
    from utils import read_table
    polyphone_file = sys.argv[1]
    phone_dict_file = sys.argv[2]
    prosody_file = sys.argv[3]
    prosody_dict_file = sys.argv[4]
    phone_dict = read_table(phone_dict_file)
    prosody_dict = read_table(prosody_dict_file)
    train_data = FrontendDataset(polyphone_file=polyphone_file,
                                 phone_dict=phone_dict,
                                 prosody_file=prosody_file,
                                 prosody_dict=prosody_dict)
    train_dataloader = DataLoader(train_data,
                                  batch_size=2,
                                  shuffle=True,
                                  collate_fn=collote_fn)
    for i, x in enumerate(train_dataloader):
        print(x)
        if i > 5:
            break
