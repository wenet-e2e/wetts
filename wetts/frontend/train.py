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
import sys
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from dataset import IGNORE_ID, FrontendDataset, collate_fn
from model import FrontendModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from utils import read_table


def compute_accuracy(logits, target):
    pred = logits.argmax(-1)
    mask = target != IGNORE_ID
    numerator = torch.sum(
        pred.masked_select(mask) == target.masked_select(mask))
    denominator = torch.sum(mask)
    if denominator > 0:
        return float(numerator) / float(denominator)
    else:
        return float("nan")


def train_or_cv(
    model,
    dataloader,
    device,
    epoch: int = 0,
    is_train=True,
    log_interval: int = 10,
    optimizer=None,
    lr_scheduler=None,
    polyphone_weight: float = 0.5,
):
    if is_train:
        context = nullcontext
        model.train()
        tag = "TRAIN"
    else:
        context = torch.no_grad
        model.eval()
        tag = "CV"
    with context():
        for batch, (inputs, phone_labels,
                    prosody_labels) in enumerate(dataloader):
            inputs["input_ids"] = inputs["input_ids"].to(device)
            inputs["token_type_ids"] = inputs["token_type_ids"].to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(device)
            phone_labels = phone_labels.to(device)
            prosody_labels = prosody_labels.to(device)
            # Forward
            phone_logits, prosody_logits = model(inputs)
            # Compute loss
            phone_loss = F.cross_entropy(phone_logits.permute(0, 2, 1),
                                         phone_labels,
                                         ignore_index=IGNORE_ID)
            phone_acc = compute_accuracy(phone_logits, phone_labels)

            prosody_loss = F.cross_entropy(prosody_logits.permute(0, 2, 1),
                                           prosody_labels,
                                           ignore_index=IGNORE_ID)
            prosody_acc = compute_accuracy(prosody_logits, prosody_labels)

            loss = polyphone_weight * phone_loss + (
                1 - polyphone_weight) * prosody_loss

            if batch % log_interval == 0:
                logstr = "Epoch {} [{}] progress {}/{} loss {:.6f}".format(
                    epoch, tag, batch, len(dataloader), loss.item())
                logstr += " polyphone_loss {:.6f} polyphone_acc {:.6f}".format(
                    phone_loss, phone_acc)
                logstr += " prosody_loss {:.6f} prosody_acc {:.6f}".format(
                    prosody_loss, prosody_acc)
                print(logstr)
            sys.stdout.flush()

            if is_train:
                # Update parameter
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()


def get_args():
    parser = argparse.ArgumentParser(description="training your network")
    parser.add_argument("--polyphone_dict",
                        required=True,
                        help="polyphone dict")
    parser.add_argument("--prosody_dict", required=True, help="prosody dict")
    parser.add_argument("--train_polyphone_data",
                        default=True,
                        help="train data file")
    parser.add_argument("--cv_polyphone_data",
                        required=True,
                        help="cv data file")
    parser.add_argument("--train_prosody_data",
                        required=True,
                        help="train data file")
    parser.add_argument("--cv_prosody_data",
                        required=True,
                        help="cv data file")
    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="gpu id for this local rank, -1 for cpu")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=4,
                        help="num training epochs")
    parser.add_argument("--log_interval",
                        type=int,
                        default=1,
                        help="log interval")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument("--polyphone_weight",
                        type=float,
                        default=0.5,
                        help="polyphone task weight")
    parser.add_argument("--model_dir", required=True, help="save model dir")
    parser.add_argument("--bert_name_or_path",
                        default='bert-chinese-base',
                        help="bert init model")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    polyphone_dict = read_table(args.polyphone_dict)
    prosody_dict = read_table(args.prosody_dict)
    num_polyphones = len(polyphone_dict)
    num_prosody = len(prosody_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name_or_path)
    collate_fn_param = partial(collate_fn, tokenizer=tokenizer)
    train_data = FrontendDataset(tokenizer, args.train_polyphone_data,
                                 polyphone_dict, args.train_prosody_data,
                                 prosody_dict)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn_param)
    cv_data = FrontendDataset(tokenizer, args.cv_polyphone_data,
                              polyphone_dict, args.cv_prosody_data,
                              prosody_dict)
    cv_dataloader = DataLoader(cv_data,
                               batch_size=args.batch_size,
                               shuffle=False,
                               collate_fn=collate_fn_param)
    # Init model
    model = FrontendModel(num_polyphones, num_prosody, args.bert_name_or_path)
    print(model)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "mps")
    print(device)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_epochs * len(train_dataloader),
    )

    # Train loop
    for i in range(args.num_epochs):
        print("Epoch {}/{}...".format(i, args.num_epochs))
        train_or_cv(
            model,
            train_dataloader,
            device,
            epoch=i,
            is_train=True,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            log_interval=args.log_interval,
            polyphone_weight=args.polyphone_weight,
        )
        train_or_cv(
            model,
            cv_dataloader,
            device,
            epoch=i,
            is_train=False,
            log_interval=args.log_interval,
            polyphone_weight=args.polyphone_weight,
        )
        torch.save(model.state_dict(),
                   os.path.join(args.model_dir, "{}.pt".format(i)))


if __name__ == "__main__":
    main()
