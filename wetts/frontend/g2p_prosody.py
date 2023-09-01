# Copyright (c) 2022, Yongqiang Li (yongqiangli@alumni.hust.edu.cn)
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

import onnxruntime as ort
from transformers import AutoTokenizer

from hanzi2pinyin import Hanzi2Pinyin


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


def get_args():
    parser = argparse.ArgumentParser(description="training your network")
    parser.add_argument("--text", required=True, help="input text")
    parser.add_argument("--hanzi2pinyin_file", required=True, help="pinyin dict")
    parser.add_argument("--polyphone_file", required=True, help="polyphone dict")
    parser.add_argument(
        "--polyphone_prosody_model", required=True, help="checkpoint model"
    )
    args = parser.parse_args()
    return args


class Frontend(object):
    def __init__(
        self,
        hanzi2pinyin_file: str,
        polyphone_prosody_model: str,
        polyphone_file: str,
    ):
        self.hanzi2pinyin = Hanzi2Pinyin(hanzi2pinyin_file)
        self.ppm_sess = ort.InferenceSession(polyphone_prosody_model)
        self.polyphone_dict = []
        self.polyphone_character_dict = []
        with open(polyphone_file) as pp_f:
            for line in pp_f.readlines():
                self.polyphone_dict.append(line.strip())

    def g2p(self, x):
        # polyphone disambiguation & prosody prediction
        tokens = tokenizer(
            list(x),
            is_split_into_words=True,
            return_tensors="np",
        )["input_ids"]
        ort_inputs = {"input": tokens}
        ort_outs = self.ppm_sess.run(None, ort_inputs)
        polyphone_pred = ort_outs[0].argmax(-1)[0][1:-1]
        prosody_pred = ort_outs[1].argmax(-1)[0][1:-1]
        pinyin = []
        for i, char in enumerate(x):
            prons = self.hanzi2pinyin.get(char)
            if len(prons) > 1:
                pinyin.append(self.polyphone_dict[polyphone_pred[i]])
            else:
                pinyin.append(prons[0])
        return pinyin, prosody_pred


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    frontend = Frontend(
        args.hanzi2pinyin_file, args.polyphone_prosody_model, args.polyphone_file
    )
    pinyin, prosody = frontend.g2p(args.text)
    print("text: {} \npinyin {} \nprosody {}".format(args.text, pinyin, prosody))


if __name__ == "__main__":
    main()
