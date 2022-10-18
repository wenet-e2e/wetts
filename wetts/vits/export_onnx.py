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
import sys

import torch

from models import SynthesizerTrn
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
    parser = argparse.ArgumentParser(description='export onnx model')
    parser.add_argument('--checkpoint', required=True, help='checkpoint')
    parser.add_argument('--cfg', required=True, help='config file')
    parser.add_argument('--onnx_model', required=True, help='onnx model')
    parser.add_argument('--phone', required=True, help='input phone dict')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    hps = utils.get_hparams_from_file(args.cfg)
    with open(args.phone) as p_f:
        phone_num = len(p_f.readlines()) + 1

    net_g = SynthesizerTrn(
        phone_num , hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length, **hps.model)
    utils.load_checkpoint(args.checkpoint, net_g, None)
    net_g.forward = net_g.export_forward
    net_g.eval()

    seq = torch.randint(low=0, high=phone_num, size=(1, 10), dtype=torch.long)
    seq_len = torch.IntTensor([seq.size(1)]).long()
    scales = torch.FloatTensor([0.667, 1.0, 0.8])

    dummy_input = (seq, seq_len, scales)
    torch.onnx.export(model=net_g,
                      args=dummy_input,
                      f=args.onnx_model,
                      input_names=['input', 'input_lengths', 'scales'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {
                              1: 'phonemes'
                          },
                          'output': {
                              1: 'audio'
                          }
                      },
                      opset_version=13,
                      verbose=False)

    # Verify onnx precision 
    torch_output = net_g(seq, seq_len, scales)
    ort_sess = ort.InferenceSession(args.onnx_model)
    ort_inputs = {'input': to_numpy(seq),
                  'input_lengths': to_numpy(seq_len),
                  'scales': to_numpy(scales)}
    onnx_output = ort_sess.run(None, ort_inputs)

if __name__ == '__main__':
    main()
