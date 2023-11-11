# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

import os

import onnxruntime as ort
import numpy as np


class Frontend:
    def __init__(self, model_dir: str):
        self.session = ort.InferenceSession(
            os.path.join(model_dir, 'final.onnx'))
        self.token2id = self.read_list(os.path.join(model_dir, 'vocab.txt'))
        self.polyphone2id = self.read_list(
            os.path.join(model_dir, 'lexicon', 'polyphone.txt'))
        self.id2polyphone = {v: k for k, v in self.polyphone2id.items()}
        self.char2pinyins = self.read_char2pinyins(
            os.path.join(model_dir, 'lexicon', 'pinyin_dict.txt'))
        self.pinyin2phones = self.read_pinyin2phones(
            os.path.join(model_dir, 'lexicon', 'lexicon.txt'))

    def read_list(self, fname: str):
        table = {}
        with open(fname) as fin:
            for i, line in enumerate(fin):
                table[line.strip()] = i
        return table

    def read_char2pinyins(self, fname: str):
        table = {}
        with open(fname) as fin:
            for line in fin:
                arr = line.split()
                assert len(arr) == 2
                char, pinyins = arr[0], arr[1].split(',')
                table[char] = pinyins
        return table

    def read_pinyin2phones(self, fname: str):
        table = {}
        with open(fname) as fin:
            for line in fin:
                arr = line.split()
                assert len(arr) >= 2
                pinyin, phones = arr[0], arr[1:]
                table[pinyin] = phones
        return table

    def compute(self, text: str):
        # TODO(Binbin Zhang): Support English/Mix Code
        tokens = ['[CLS]'] + [str(x) for x in text] + ['[SEP]']
        token_ids = [self.token2id[x] for x in tokens]
        outputs = self.session.run(
            None, {'input': np.expand_dims(np.array(token_ids), axis=0)})
        pinyin_prob, prosody_prob = outputs[0][0], outputs[1][0]
        pinyins = []
        for i in range(1, len(tokens) - 1):
            x = tokens[i]
            arr = self.char2pinyins[x]
            if len(arr) > 1:
                poly_probs = [
                    pinyin_prob[i][self.polyphone2id[p]] for p in arr
                ]
                max_idx = poly_probs.index(max(poly_probs))
                pinyins.append(arr[max_idx])
            else:
                pinyins.append(arr[0])
        prosodys = prosody_prob.argmax(axis=1).tolist()
        outputs = ['sil']
        for i in range(len(pinyins)):
            outputs.extend(self.pinyin2phones[pinyins[i]])
            outputs.append('#{}'.format(prosodys[i]))
        outputs[-1] = '#4'
        return outputs
