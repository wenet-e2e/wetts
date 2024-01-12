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
from pathlib import Path

import torch

from model.models import SynthesizerTrn
from utils import task


def get_args():
    parser = argparse.ArgumentParser(description="export onnx model")
    parser.add_argument("--checkpoint", required=True, help="checkpoint")
    parser.add_argument("--cfg", required=True, help="config file")
    parser.add_argument("--onnx_model", required=True, help="onnx model")
    parser.add_argument("--phone_table",
                        required=True,
                        help="input phone dict")
    parser.add_argument("--speaker_table", default=None, help="speaker table")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="export streaming model"
    )
    parser.add_argument(
        "--providers",
        required=False,
        default="CPUExecutionProvider",
        choices=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="the model to send request to",
    )
    args = parser.parse_args()
    return args

def add_prefix(filepath, prefix):
    filepath = Path(filepath)
    return str(filepath.parent / (prefix + filepath.name))


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    hps = task.get_hparams_from_file(args.cfg)
    hps['model']['is_onnx'] = True

    phone_num = len(open(args.phone_table).readlines())
    num_speakers = len(open(args.speaker_table).readlines())

    posterior_channels = hps.data.filter_length // 2 + 1
    if ("use_mel_posterior_encoder" in hps.model.keys()
            and hps.model.use_mel_posterior_encoder):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = hps.data.n_mel_channels  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")

    net_g = SynthesizerTrn(phone_num,
                           posterior_channels,
                           hps.train.segment_size // hps.data.hop_length,
                           n_speakers=num_speakers,
                           **hps.model)
    task.load_checkpoint(args.checkpoint, net_g, None)
    if hasattr(net_g.flow, 'remove_weight_norm'):
        net_g.flow.remove_weight_norm()
    net_g.dec.remove_weight_norm()
    net_g.forward = net_g.export_forward
    net_g.eval()

    seq = torch.randint(low=0, high=phone_num, size=(1, 10), dtype=torch.long)
    seq_len = torch.IntTensor([seq.size(1)]).long()
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    # make triton dynamic shape happy
    scales = scales.unsqueeze(0)
    sid = torch.IntTensor([0]).long()
    z = torch.randn(1, hps.model.inter_channels, 105)
    g = torch.randn(1, hps.model.gin_channels, 1)

    if args.streaming:
        net_g.forward = net_g.export_encoder_forward
        dummy_input = (seq, seq_len, scales, sid)
        torch.onnx.export(
            model=net_g,
            args=dummy_input,
            f=add_prefix(args.onnx_model, 'encoder_'),
            input_names=["input", "input_lengths", "scales", "sid"],
            output_names=["z", "g"],
            dynamic_axes={
                "input": {
                    0: "batch",
                    1: "phonemes"
                },
                "input_lengths": {
                    0: "batch"
                },
                "scales": {
                    0: "batch"
                },
                "sid": {
                    0: "batch"
                },
                "z": {0: "batch", 2: "L"},
                "g": {0: "batch"},
            },
            opset_version=13,
            verbose=False,
        )
        net_g.forward = net_g.export_decoder_forward
        dummy_input = (z, g)
        torch.onnx.export(
            model=net_g,
            args=dummy_input,
            f=add_prefix(args.onnx_model, 'decoder_'),
            input_names=["z", "g"],
            output_names=["output"],
            dynamic_axes={
                "z": {0: "batch", 2: "L"},
                "g": {0: "batch"},
                "output": {
                    0: "batch",
                    1: "audio",
                    2: "audio_length"
                },
            },
            opset_version=13,
            verbose=False,
        )
    else:
        dummy_input = (seq, seq_len, scales, sid)
        torch.onnx.export(
            model=net_g,
            args=dummy_input,
            f=args.onnx_model,
            input_names=["input", "input_lengths", "scales", "sid"],
            output_names=["output"],
            dynamic_axes={
                "input": {
                    0: "batch",
                    1: "phonemes"
                },
                "input_lengths": {
                    0: "batch"
                },
                "scales": {
                    0: "batch"
                },
                "sid": {
                    0: "batch"
                },
                "output": {
                    0: "batch",
                    1: "audio",
                    2: "audio_length"
                },
            },
            opset_version=13,
            verbose=False,
        )


if __name__ == "__main__":
    main()
