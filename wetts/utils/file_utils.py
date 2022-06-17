#!/usr/bin/env python3
# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com), Jie Chen
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

import torch

def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_scp(scp_file):
    lists = []
    with open(scp_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            lists.append(arr)
    return lists


def read_key2id(scp_file):
    lists = read_scp(scp_file)
    key2id = {x[0]: int(x[1]) for x in lists}
    return key2id


def read_lexicon(lexicon_file):
    lines = read_lists(lexicon_file)
    lexicon = {}
    for line in lines:
        tokens = line.split()
        lexicon[tokens[0]] = tokens[1:]
    return lexicon


def load_ckpt(path):
    with open(path, 'rb') as fin:
        return torch.load(fin, map_location='cpu')
