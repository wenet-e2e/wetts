# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com), Jie Chen
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
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from yacs import config

from wetts.models.am.fastspeech2.fastspeech2 import FastSpeech2
from wetts.models.am.fastspeech2.module import loss
from wetts.models.am.fastspeech2.module.dataset import (
    FastSpeech2TrainingDataset)
from wetts.utils.lr_scheduler import transformer_lr_scheduler
from wetts.utils.plot import plot_mel

train_step = 1
val_step = 1


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='number of workers for dataloader')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data_list',
                        required=True,
                        help='training data list file')
    parser.add_argument('--val_data_list',
                        required=True,
                        help='validation data list file')
    parser.add_argument('--phn2id_file', required=True, help='phn2id file')
    parser.add_argument('--spk2id_file', required=True, help='spk2id file')
    parser.add_argument('--special_tokens_file',
                        required=True,
                        help='special tokens file')
    parser.add_argument('--cmvn_dir', required=True, help='cmvn dir')
    parser.add_argument('--ckpt', help='path to ckpt to resume training')
    args = parser.parse_args(argv)
    return args


def train(epoch, model, data_loader, loss_fn, optimizer, lr_scheduler,
          summary_writer):
    global train_step
    model.train()
    for (keys, speakers, durations, text, mel, pitch, energy, text_length,
         mel_length, token_types) in data_loader:
        optimizer.zero_grad()
        speakers = speakers.cuda()
        durations = durations.cuda()
        text = text.cuda()
        mel = mel.cuda()
        pitch = pitch.cuda()
        energy = energy.cuda()
        text_length = text_length.cuda()
        mel_length = mel_length.cuda()
        token_types = token_types.cuda()
        (mel_prediction, postnet_mel_prediction, mel_mask, pitch_prediction,
         energy_prediction, log_duration_prediction,
         enc_output_mask) = model(text,
                                  text_length,
                                  token_types,
                                  duration_target=durations,
                                  pitch_target=pitch,
                                  energy_target=energy,
                                  speaker=speakers)
        mel_mask = mel_mask.unsqueeze(2)
        (duration_loss, pitch_loss, energy_loss, mel_loss,
         postnet_mel_loss) = loss_fn(durations, log_duration_prediction, pitch,
                                     pitch_prediction, energy,
                                     energy_prediction, enc_output_mask, mel,
                                     mel_prediction, postnet_mel_prediction,
                                     mel_mask)
        total_loss = (duration_loss + pitch_loss + energy_loss + mel_loss +
                      postnet_mel_loss)
        total_loss.backward()
        summary_writer.add_scalar('train duration loss', duration_loss.item(),
                                  train_step)
        summary_writer.add_scalar('train pitch loss', pitch_loss.item(),
                                  train_step)
        summary_writer.add_scalar('train energy loss', energy_loss.item(),
                                  train_step)
        summary_writer.add_scalar('train mel loss', mel_loss.item(),
                                  train_step)
        summary_writer.add_scalar('train postnet mel loss',
                                  postnet_mel_loss.item(), train_step)
        summary_writer.add_scalar('train total loss', total_loss.item(),
                                  train_step)
        summary_writer.add_scalar('learning rate',
                                  lr_scheduler.get_last_lr()[0], train_step)
        print(("epoch {}, train step {}, train duration loss {}, "
               "train pitch loss {}, train energy loss {}, train mel loss {}, "
               "train postnet mel loss {},  train total loss: {}").format(
                   epoch + 1, train_step, duration_loss.item(),
                   pitch_loss.item(), energy_loss.item(), mel_loss.item(),
                   postnet_mel_loss.item(), total_loss.item()))
        optimizer.step()
        lr_scheduler.step()
        train_step += 1


def eval(epoch, model, data_loader, loss_fn, summary_writer):
    global val_step
    model.eval()
    with torch.no_grad():
        for (keys, speakers, durations, text, mel, pitch, energy, text_length,
             mel_length, token_types) in data_loader:
            speakers = speakers.cuda()
            durations = durations.cuda()
            text = text.cuda()
            mel = mel.cuda()
            pitch = pitch.cuda()
            energy = energy.cuda()
            text_length = text_length.cuda()
            mel_length = mel_length.cuda()
            token_types = token_types.cuda()
            (mel_prediction, postnet_mel_prediction, mel_mask,
             pitch_prediction, energy_prediction, log_duration_prediction,
             enc_output_mask) = model(text,
                                      text_length,
                                      token_types,
                                      duration_target=durations,
                                      pitch_target=pitch,
                                      energy_target=energy,
                                      speaker=speakers)
            mel_mask = mel_mask.unsqueeze(2)
            (duration_loss, pitch_loss, energy_loss, mel_loss,
             postnet_mel_loss) = loss_fn(durations, log_duration_prediction,
                                         pitch, pitch_prediction, energy,
                                         energy_prediction, enc_output_mask,
                                         mel, mel_prediction,
                                         postnet_mel_prediction, mel_mask)
            total_loss = (duration_loss + pitch_loss + energy_loss + mel_loss +
                          postnet_mel_loss)
            summary_writer.add_scalar('eval duration loss',
                                      duration_loss.item(), val_step)
            summary_writer.add_scalar('eval pitch loss', pitch_loss.item(),
                                      val_step)
            summary_writer.add_scalar('eval energy loss', energy_loss.item(),
                                      val_step)
            summary_writer.add_scalar('eval mel loss', mel_loss.item(),
                                      val_step)
            summary_writer.add_scalar('eval postnet mel loss',
                                      postnet_mel_loss.item(), val_step)
            summary_writer.add_scalar('eval total loss', total_loss.item(),
                                      val_step)
            print(
                ("epoch {}, eval step {}, eval duration loss {}, "
                 "eval pitch loss {}, eval energy loss {}, eval mel loss {}, "
                 "eval postnet mel loss {}, eval total loss: {}").format(
                     epoch + 1, val_step, duration_loss.item(),
                     pitch_loss.item(), energy_loss.item(), mel_loss.item(),
                     postnet_mel_loss.item(), total_loss.item()))
            val_step += 1
        mels_to_plot = [
            mel[0, :~mel_mask[0].sum()].cpu(),
            mel_prediction[0, :~mel_mask[0].sum()].cpu(),
            postnet_mel_prediction[0, :~mel_mask[0].sum()].cpu()
        ]
        summary_writer.add_figure(
            'eval mels',
            plot_mel(mels_to_plot,
                     ['ground truth', 'prediction', 'postnet prediction']),
            epoch + 1)


def save_ckpt(path, model, lr_scheduler, optimizer, train_step, val_step,
              epoch):
    torch.save([
        model.state_dict(),
        lr_scheduler.state_dict(),
        optimizer.state_dict(), train_step, val_step, epoch
    ], path)


def load_ckpt(path):
    with open(path, 'rb') as fin:
        (model_state_dict, lr_scheduler_state_dict, optimizer_state_dict,
         train_step, val_step, epoch) = torch.load(fin, 'cpu')
        return (model_state_dict, lr_scheduler_state_dict,
                optimizer_state_dict, train_step, val_step, epoch)


def main(args):
    global train_step, val_step
    with open(args.config, 'r') as fin:
        conf = config.load_cfg(fin)

    os.makedirs(conf.log_dir.checkpoint, exist_ok=True)
    (train_dataset, _, pitch_stats, energy_stats,
     phn2id) = FastSpeech2TrainingDataset(args.train_data_list,
                                          args.spk2id_file, args.phn2id_file,
                                          args.special_tokens_file,
                                          args.cmvn_dir, conf)
    val_dataset, *_ = FastSpeech2TrainingDataset(args.val_data_list,
                                                 args.spk2id_file,
                                                 args.phn2id_file,
                                                 args.special_tokens_file,
                                                 args.cmvn_dir, conf)
    pitch_mean, pitch_sigma, pitch_min, pitch_max = pitch_stats
    energy_mean, energy_sigma, energy_min, energy_max = energy_stats

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   num_workers=args.num_workers)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=None,
                                 num_workers=conf.num_workers)
    model = FastSpeech2(
        conf.model.d_model, conf.model.n_enc_layer, conf.model.n_enc_head,
        conf.model.n_enc_conv_filter, conf.model.enc_conv_kernel_size,
        conf.model.enc_dropout, len(phn2id), conf.model.padding_idx,
        conf.model.n_va_conv_filter, conf.model.va_conv_kernel_size,
        conf.model.va_dropout, pitch_min, pitch_max, pitch_mean, pitch_sigma,
        energy_min, energy_max, energy_mean, energy_sigma,
        conf.model.n_pitch_bin, conf.model.n_energy_bin,
        conf.model.n_dec_layer, conf.model.n_dec_head,
        conf.model.n_dec_conv_filter, conf.model.dec_conv_kernel_size,
        conf.model.dec_dropout, conf.n_mels, conf.n_speaker,
        conf.model.postnet_kernel_size, conf.model.postnet_hidden_dim,
        conf.model.n_postnet_conv_layers, conf.model.postnet_dropout,
        conf.model.max_pos_enc_len)
    model = model.cuda()

    loss_fn = loss.FastSpeech2Loss()

    optimizer = Adam(model.parameters(), 1, conf.optimizer.betas,
                     conf.optimizer.eps)
    lr_scheduler = transformer_lr_scheduler(optimizer, conf.model.d_model,
                                            conf.optimizer.warmup_steps)
    writer = SummaryWriter(conf.log_dir.tensorboard)
    last_epoch = 0
    if args.ckpt:
        (model_state_dict, lr_scheduler_state_dict, optimizer_state_dict,
         train_step, val_step, last_epoch) = load_ckpt(args.ckpt)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        print("Resume training from epoch {}.".format(last_epoch))

    for epoch in range(last_epoch, conf.epoch):
        train(epoch, model, train_data_loader, loss_fn, optimizer,
              lr_scheduler, writer)
        eval(epoch, model, val_data_loader, loss_fn, writer)
        save_ckpt(
            os.path.join(conf.log_dir.checkpoint,
                         'fastspeech2_{}.ckpt'.format(epoch + 1)), model,
            lr_scheduler, optimizer, train_step, val_step, epoch + 1)


if __name__ == '__main__':
    main(get_args())
