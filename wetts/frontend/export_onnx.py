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

import argparse
import os

import torch
import onnxruntime as ort

from model import FrontendModel
from utils import read_table


def get_args():
    parser = argparse.ArgumentParser(description="export onnn model")
    parser.add_argument("--polyphone_dict", required=True, help="polyphone dict file")
    parser.add_argument("--prosody_dict", required=True, help="train data file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--onnx_model", required=True, help="onnx model path")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    polyphone_dict = read_table(args.polyphone_dict)
    prosody_dict = read_table(args.prosody_dict)
    num_polyphones = len(polyphone_dict)
    num_prosody = len(prosody_dict)

    # Init model
    model = FrontendModel(num_polyphones, num_prosody)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.forward = model.export_forward
    model.eval()

    dummy_input = torch.ones(1, 10, dtype=torch.int64)
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_model,
        input_names=["input"],
        output_names=["polyphone_output", "prosody_output"],
        dynamic_axes={
            "input": {1: "T"},
            "polyphone_output": {1: "T"},
            "prosody_output": {1: "T"},
        },
        opset_version=13,
        verbose=False,
    )

    # Verify onnx precision
    torch_output = model(dummy_input)
    ort_sess = ort.InferenceSession(args.onnx_model)
    onnx_output = ort_sess.run(None, {"input": dummy_input.numpy()})
    print(torch_output[1])
    print(onnx_output[1])
    if torch.allclose(
        torch_output[0], torch.tensor(onnx_output[0]), atol=1e-3
    ) and torch.allclose(torch_output[1], torch.tensor(onnx_output[1]), atol=1e-3):
        print("Export to onnx succeed!")
    else:
        print(
            """Export to onnx succeed, but pytorch/onnx have different
                 outputs when given the same input, please check!!!"""
        )


if __name__ == "__main__":
    main()
