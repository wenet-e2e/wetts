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
import random

import onnxruntime as ort
import torch
import torch.nn.functional as F
from model import FrontendModel
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer
from utils import read_table


def cosine_similarity(tensor1, tensor2):
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    similarity = F.cosine_similarity(flat1.unsqueeze(0),
                                     flat2.unsqueeze(0),
                                     dim=1)
    return similarity.item()


def get_args():
    parser = argparse.ArgumentParser(description="export onnn model")
    parser.add_argument("--polyphone_dict",
                        required=True,
                        help="polyphone dict file")
    parser.add_argument("--prosody_dict",
                        required=True,
                        help="train data file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--onnx_model", required=True, help="onnx model path")
    parser.add_argument("--quant_onnx_model", help="quant onnx model path")
    parser.add_argument("--bert_name_or_path",
                        default='bert-chinese-base',
                        help="bert init model")
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
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name_or_path)
    model = FrontendModel(num_polyphones, num_prosody, args.bert_name_or_path)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"),
                          strict=False)
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
            "input": {
                1: "T"
            },
            "polyphone_output": {
                1: "T"
            },
            "prosody_output": {
                1: "T"
            },
        },
        opset_version=13,
        verbose=False,
    )

    # Verify onnx precision
    rand_len = random.randint(5, 20)
    rand_input = torch.randint(0, 100, (1, rand_len))
    print('rand input', rand_input)
    torch_output = model(rand_input)
    print(torch_output[0].size(), torch_output[1].size())

    def verify_export(onnx_model):
        ort_sess = ort.InferenceSession(onnx_model)
        onnx_output = ort_sess.run(None, {"input": rand_input.numpy()})
        sim = cosine_similarity(torch_output[1],
                                torch.from_numpy(onnx_output[1]))
        print(f"Export to onnx {onnx_model}, similarity {sim}")

    verify_export(args.onnx_model)
    if args.quant_onnx_model is not None:
        quantize_dynamic(args.onnx_model,
                         args.quant_onnx_model,
                         weight_type=QuantType.QUInt8)
        verify_export(args.quant_onnx_model)


if __name__ == "__main__":
    main()
