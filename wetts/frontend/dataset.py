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
from torch.utils.data import Dataset

IGNORE_ID = -100


class FrontendDataset(Dataset):

    def __init__(
        self,
        tokenizer=None,
        polyphone_file=None,
        polyphone_dict=None,
        prosody_file=None,
        prosody_dict=None,
    ):
        self.tokenizer = tokenizer
        self.data = []
        self.polyphone_dict = polyphone_dict
        self.prosody_dict = prosody_dict
        if polyphone_file is not None and polyphone_dict is not None:
            data = self.read_polyphone_data(polyphone_file)
            print("Reading {} polyphone data".format(len(data)))
            self.data += data

        if prosody_file is not None and prosody_dict is not None:
            data = self.read_prosody_data(prosody_file)
            print("Reading {} prosody data".format(len(data)))
            self.data += data

    def read_polyphone_data(self, polyphone_file):
        """If there is a polyphone, it is surrounded by ▁, such as
        ```
        宋代出现了▁le5▁燕乐音阶的记载
        爆发了▁le5▁占领华尔街示威活动
        ```
        """
        data = []
        with open(polyphone_file, encoding="utf8") as fin:
            for line in fin:
                arr = line.strip().strip("▁").split("▁")
                # Check
                tokens = []
                all_polyphones = []
                for i in range(0, len(arr), 2):
                    toks = self.tokenizer.encode(arr[i],
                                                 add_special_tokens=False)
                    polyphones = [IGNORE_ID] * len(toks)
                    if i + 1 < len(arr):
                        if arr[i + 1] in self.polyphone_dict:
                            polyphones[-1] = self.polyphone_dict[arr[i + 1]]
                        # else:
                        #     print("Skip unknown polyphone {}".format(arr[i + 1]))
                    all_polyphones.extend(polyphones)
                    tokens.append(arr[i])
                data.append({
                    "sentence": tokens,
                    "polyphones": all_polyphones,
                    "prosody": [IGNORE_ID] * len(all_polyphones),
                })
        return data

    def read_prosody_data(self, prosody_file):
        """Each line of prosody is like:
        ```
        今天 #1 天气 #1 怎么样 #3
        ```
        """
        data = []
        with open(prosody_file, "r", encoding="utf8") as fin:
            for line in fin:
                arr = line.strip().split()
                # Check
                ok = True
                if len(arr) % 2 != 0:
                    ok = False
                else:
                    for i in range(0, len(arr), 2):
                        # #N and N is less than `prosody_dict`
                        if (arr[i + 1][0] != "#"
                                or not arr[i + 1][1:].encode("utf8").isdigit()
                                or int(arr[i + 1][1:]) >= len(
                                    self.prosody_dict)):
                            ok = False
                            break
                if not ok:
                    print("Ignore line {}".format(line.strip()))
                    continue

                tokens = []
                prosody = []
                for i in range(0, len(arr), 2):
                    toks = self.tokenizer.encode(arr[i],
                                                 add_special_tokens=False)
                    rank = int(arr[i + 1][1:])
                    rhythm = [0] * len(toks)
                    rhythm[-1] = rank
                    prosody.extend(rhythm)
                    tokens.append(arr[i])
                data.append({
                    "sentence": tokens,
                    "polyphones": [IGNORE_ID] * len(prosody),
                    "prosody": prosody,
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch_samples, tokenizer):
    batch_sentence, batch_polyphones, batch_prosody = [], [], []
    for sample in batch_samples:
        batch_sentence.append(sample["sentence"])
        batch_polyphones.append(sample["polyphones"])
        batch_prosody.append(sample["prosody"])
    batch_inputs = tokenizer(
        batch_sentence,
        padding=True,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    polyphone_label = np.ones_like(batch_inputs["input_ids"],
                                   dtype=int) * IGNORE_ID
    for idx, polyphones in enumerate(batch_polyphones):
        polyphone_label[idx][1:len(polyphones) + 1] = np.array(polyphones,
                                                               dtype=np.int32)
    prosody_label = np.ones_like(batch_inputs["input_ids"],
                                 dtype=int) * IGNORE_ID
    for idx, prosody in enumerate(batch_prosody):
        prosody_label[idx][1:len(prosody) + 1] = np.array(prosody,
                                                          dtype=np.int32)

    return batch_inputs, torch.tensor(polyphone_label), torch.tensor(
        prosody_label)
