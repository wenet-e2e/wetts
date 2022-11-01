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

import numpy as np
from scipy.io import wavfile
import torch

import commons
import utils


try:
    import onnxruntime as ort
except ImportError:
    print('Please install onnxruntime!')
    sys.exit(1)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad \
        else tensor.detach().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--onnx_model', required=True, help='onnx model')
    parser.add_argument('--cfg', required=True, help='config file')
    parser.add_argument('--outdir', required=True, help='ouput directory')
    parser.add_argument('--phone', required=True, help='input phone dict')
    parser.add_argument('--test_file', required=True, help='test file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    phone_dict = {}
    with open(args.phone) as p_f:
        for line in p_f:
            phone_id = line.strip().split()
            phone_dict[phone_id[0]] = int(phone_id[1])
    hps = utils.get_hparams_from_file(args.cfg)

    ort_sess = ort.InferenceSession(args.onnx_model)
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    # make triton dynamic shape happy
    scales = scales.unsqueeze(0)

    with open(args.test_file) as fin:
        for line in fin:
            audio_text = line.strip().split("|")
            seq = [phone_dict[symbol] for symbol in audio_text[1].split()]
            if hps.data.add_blank:
                seq = commons.intersperse(seq, 0)
            x = torch.LongTensor([seq])
            x_len = torch.IntTensor([x.size(1)]).long()
            ort_inputs = {'input': to_numpy(x),
                          'input_lengths': to_numpy(x_len),
                          'scales': to_numpy(scales)}
            audio = np.squeeze(ort_sess.run(None, ort_inputs))
            audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
            audio = np.clip(audio, -32767.0, 32767.0)
            wavfile.write(args.outdir + "/" + audio_text[0].split("/")[-1],
                          hps.data.sampling_rate, audio.astype(np.int16))

if __name__ == '__main__':
    main()
