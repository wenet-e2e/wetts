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

from transformers import AutoTokenizer
import zhconv

from cn_tn import TextNorm
from hanzi2pinyin import Hanzi2Pinyin


try:
    import onnxruntime as ort
except ImportError:
    print('Please install onnxruntime!')
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--ips', required=True, help='input text')
    parser.add_argument('--hanzi2pinyin_file', required=True, help='input ')
    parser.add_argument('--polyphone_phone_file', required=True, help='input ')
    parser.add_argument('--polyphone_character_file', required=True, help='input ')
    parser.add_argument('--polyphone_prosody_model', required=True, help='checkpoint model')
    args = parser.parse_args()
    return args


class Frontend(object):
    def __init__(
        self,
        hanzi2pinyin_file: str,
        polyphone_prosody_model: str,
        polyphone_phone_file: str,
        polyphone_character_file: str,
    ):
        self.tn = TextNorm(remove_space=True)
        self.polyphone_prosody_model = polyphone_prosody_model
        self.hanzi2pinyin = Hanzi2Pinyin(hanzi2pinyin_file)
        self.polyphone_phone_dict = []
        self.polyphone_character_dict = []
        with open(polyphone_phone_file) as pp_f:
            for line in pp_f.readlines():
                self.polyphone_phone_dict.append(line.strip())
        with open(polyphone_character_file) as pc_f:
            for line in pc_f.readlines():
                self.polyphone_character_dict.append(line.strip())

    def g2p(self, x):
        # tradition to simple
        x = zhconv.convert(x, 'zh-hans')
        # text normalization
        x = self.tn(x)
        # hanzi2pinyin
        pinyin = self.hanzi2pinyin.convert(x)
        # polyphone disambiguation & prosody prediction
        tokens = tokenizer(list(x),
                           is_split_into_words=True,
                           return_tensors="np")['input_ids']
        ppm_sess = ort.InferenceSession(self.polyphone_prosody_model)
        ort_inputs = {'input': tokens}
        ort_outs = ppm_sess.run(None, ort_inputs)
        polyphone_pred = ort_outs[0].argmax(-1)[0][1:-1]
        prosody_pred = ort_outs[1].argmax(-1)[0][1:-1]
        index = 0
        for char in x:
            if char in self.polyphone_character_dict:
                pinyin[index] = self.polyphone_phone_dict[polyphone_pred[index]]
            index += 1
        return pinyin, prosody_pred

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    frontend = Frontend(args.hanzi2pinyin_file,
                        args.polyphone_prosody_model,
                        args.polyphone_phone_file,
                        args.polyphone_character_file)
    pinyin, prosody = frontend.g2p(args.ips)
    print("text: {} \npinyin {} \nprosody {}".format(
          args.ips, pinyin, prosody))
if __name__ == '__main__':
    main()
