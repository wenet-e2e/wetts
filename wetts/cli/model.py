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

import numpy as np
import onnxruntime as ort

from wetts.cli.frontend import Frontend
from wetts.cli.hub import Hub


class Model:
    def __init__(self, backend_dir: str, front_dir: str):
        self.frontend = Frontend(front_dir)
        self.session = ort.InferenceSession(
            os.path.join(backend_dir, 'final.onnx'))
        self.phone2id = self.read_table(os.path.join(backend_dir,
                                                     'phones.txt'))
        self.speaker2id = self.read_table(
            os.path.join(backend_dir, 'speaker.txt'))

    def read_table(self, fname: str):
        table = {}
        with open(fname) as fin:
            for line in fin:
                arr = line.split()
                assert len(arr) == 2
                table[arr[0]] = int(arr[1])
        return table

    def synthesis(self, text: str, speaker: str = 'default'):
        phonemes = self.frontend.compute(text)
        phonemes_id = [self.phone2id[x] for x in phonemes]
        scales = [0.667, 1.0, 0.8]
        sid = self.speaker2id.get(speaker, 0)
        outputs = self.session.run(
            None, {
                'input':
                np.expand_dims(np.array(phonemes_id), axis=0),
                'input_lengths':
                np.array([len(phonemes)]),
                'scales':
                np.expand_dims(np.array(scales, dtype=np.float32), axis=0),
                'sid':
                np.array([sid]),
            })
        audio = outputs[0][0][0]
        audio = (audio * 32767).astype(np.int16)
        return phonemes, audio


def load_model():
    front_dir = Hub.get_model('frontend')
    backend_dir = Hub.get_model('multilingual')
    model = Model(backend_dir, front_dir)
    return model
