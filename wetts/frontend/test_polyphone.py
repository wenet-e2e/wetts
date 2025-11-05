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
from functools import partial

import torch
from dataset import IGNORE_ID, FrontendDataset, collate_fn
from model import FrontendModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import read_table


def get_args():
    parser = argparse.ArgumentParser(description="training your network")
    parser.add_argument("--polyphone_dict",
                        required=True,
                        help="polyphone dict file")
    parser.add_argument("--prosody_dict",
                        required=True,
                        help="train data file")
    parser.add_argument("--test_data", required=True, help="test data file")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
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

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name_or_path)
    collate_fn_param = partial(collate_fn, tokenizer=tokenizer)
    test_data = FrontendDataset(tokenizer,
                                polyphone_file=args.test_data,
                                polyphone_dict=polyphone_dict)
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn_param,
    )
    # Init model
    model = FrontendModel(num_polyphones, num_prosody, args.bert_name_or_path)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    model.eval()
    num_total = 0
    num_correct = 0
    with torch.no_grad():
        pbar = tqdm(total=len(test_dataloader))
        for _, (inputs, labels, _) in enumerate(test_dataloader):
            logits, _ = model(inputs)
            mask = labels != IGNORE_ID
            num_total += torch.sum(mask)
            pred = logits.argmax(-1)
            equal = (pred == labels) * mask
            num_correct += torch.sum(equal)
            pbar.update(1)
        pbar.close()

    print("Accuracy: {}".format(num_correct / num_total))


if __name__ == "__main__":
    main()
