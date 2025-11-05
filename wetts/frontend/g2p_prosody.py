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

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="training your network")
    parser.add_argument("--text", required=True, help="input text")
    parser.add_argument("--lexicon_file", required=True, help="pinyin dict")
    parser.add_argument("--polyphone_file",
                        required=True,
                        help="polyphone dict")
    parser.add_argument("--polyphone_prosody_model",
                        required=True,
                        help="checkpoint model")
    parser.add_argument("--bert_name_or_path",
                        default='bert-chinese-base',
                        help="bert init model")
    args = parser.parse_args()
    return args


class Frontend(object):

    def __init__(
        self,
        tokenizer_path: str,
        lexicon_file: str,
        polyphone_prosody_model: str,
        polyphone_file: str,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.ppm_sess = ort.InferenceSession(polyphone_prosody_model)
        self.lexicon = {}
        with open(lexicon_file) as f:
            for line in f:
                arr = line.strip().split(maxsplit=1)
                char, prons = arr[0], arr[1].split(',')
                self.lexicon[char] = prons
        self.polyphone_dict = []
        with open(polyphone_file) as pp_f:
            for line in pp_f.readlines():
                self.polyphone_dict.append(line.strip())

    def g2p(self, text):
        # polyphone disambiguation & prosody prediction
        tokens = self.tokenizer(
            list(text),
            is_split_into_words=True,
            return_tensors="np",
        )["input_ids"]
        ort_inputs = {"input": tokens}
        print(ort_inputs)
        ort_outs = self.ppm_sess.run(None, ort_inputs)
        prosody_pred = ort_outs[1].argmax(-1)[0][1:-1]
        pinyin = []
        for i, char in enumerate(text):
            prons = self.lexicon[char]
            if len(prons) > 1:
                polyphone_ids = []
                # The predicted probability for each pronunciation of the polyphone.
                preds = []
                for pron in prons:
                    index = self.polyphone_dict.index(pron)
                    polyphone_ids.append(index)
                    preds.append(ort_outs[0][0][i + 1][index])
                print(prons, preds)
                preds = np.array(preds)
                id = polyphone_ids[preds.argmax(-1)]
                pinyin.append(self.polyphone_dict[id])
            else:
                pinyin.append(prons[0])
        return pinyin, prosody_pred


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    frontend = Frontend(args.bert_name_or_path, args.lexicon_file,
                        args.polyphone_prosody_model, args.polyphone_file)
    pinyin, prosody = frontend.g2p(args.text)
    print("text: {} \npinyin {} \nprosody {}".format(args.text, pinyin,
                                                     prosody))


if __name__ == "__main__":
    main()
