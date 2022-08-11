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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

from dataset import ProsodyDataset, collote_fn, IGNORE_ID
from prosody_model import ProsodyModel


def compute_accuracy(logits, target):
    pred = logits.argmax(-1)
    mask = target != IGNORE_ID
    numerator = torch.sum(
        pred.masked_select(mask) == target.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


def train(model, dataloader, optimizer, lr_scheduler, device, epoch,
          log_interval):
    model.train()
    for batch, (inputs, labels) in enumerate(dataloader):
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        labels = labels.to(device)
        # Forward
        logits = model(inputs)
        # Compute loss
        loss = F.cross_entropy(logits.permute(0, 2, 1),
                               labels,
                               ignore_index=IGNORE_ID)
        acc = compute_accuracy(logits, labels)
        if batch % log_interval == 0:
            print(
                'Epoch {} [TRAIN] progress {}/{} loss {:.6f} accuracy {:.6f}'.
                format(epoch, batch, len(dataloader), loss.item(), acc))
        sys.stdout.flush()
        # Update parameter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


def cv(model, dataloader, device, epoch, log_interval):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(dataloader):
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            # Forward
            logits = model(inputs)
            # Compute loss
            loss = F.cross_entropy(logits.permute(0, 2, 1),
                                   labels,
                                   ignore_index=IGNORE_ID)
            acc = compute_accuracy(logits, labels)
            total_loss += loss.item()
            if batch % log_interval == 0:
                print(
                    'Epoch {} [CV] progress {}/{} loss {:.6f} accuracy {:.6f}'.
                    format(epoch, batch, len(dataloader), loss.item(), acc))
            sys.stdout.flush()
    return total_loss / len(dataloader)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--num_prosody',
                        type=int,
                        default=4,
                        help='num prosody classes')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=4,
                        help='num training epochs')
    parser.add_argument('--log_interval',
                        type=int,
                        default=1,
                        help='log interval')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    train_data = ProsodyDataset(args.train_data)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=collote_fn)
    cv_data = ProsodyDataset(args.cv_data)
    cv_dataloader = DataLoader(cv_data,
                               batch_size=args.batch_size,
                               shuffle=False,
                               collate_fn=collote_fn)
    # Init model
    model = ProsodyModel(args.num_prosody)
    print(model)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
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
        print('Epoch {}/{} ...'.format(i, args.num_epochs))
        train(model, train_dataloader, optimizer, lr_scheduler, device, i,
              args.log_interval)
        cv_loss = cv(model, cv_dataloader, device, i, args.log_interval)
        torch.save(model.state_dict(),
                   os.path.join(args.model_dir, '{}.pt'.format(i)))


if __name__ == '__main__':
    main()
