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
import pathlib
import random

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


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='number of workers for dataloader')
    parser.add_argument('--config',
                        required=True,
                        type=str,
                        help='config file')
    parser.add_argument('--train_data_list',
                        required=True,
                        type=str,
                        help='training data list file')
    parser.add_argument('--val_data_list',
                        required=True,
                        type=str,
                        help='validation data list file')
    parser.add_argument('--phn2id_file',
                        required=True,
                        type=str,
                        help='phn2id file')
    parser.add_argument(
        '--spk2id_file',
        type=str,
        required=True,
        help='path to spk2id file, this file must be provided '
        'for both multi-speaker FastSpeech2 and single-speaker '
        'FastSpeech2')
    parser.add_argument('--special_tokens_file',
                        required=True,
                        type=str,
                        help='special tokens file')
    parser.add_argument('--cmvn_dir', required=True, type=str, help='cmvn dir')
    parser.add_argument('--ckpt',
                        type=str,
                        help='path to ckpt to resume training')
    parser.add_argument('--log_dir',
                        default='log',
                        type=str,
                        help='path to save tensorboard log and checkpoint')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--epoch',
                        type=int,
                        default=100,
                        help='number of epoch to train')
    args = parser.parse_args(argv)
    return args


def train(epoch, model, data_loader, loss_fn, optimizer, lr_scheduler,
          summary_writer):
    global train_step
    model.train()
    for (keys, speakers, durations, text, mel, pitch, energy, text_length,
         mel_length, token_types, _) in data_loader:
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
        (mel_prediction, postnet_mel_prediction, mel_mask, mel_len,
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
                   epoch, train_step, duration_loss.item(), pitch_loss.item(),
                   energy_loss.item(), mel_loss.item(),
                   postnet_mel_loss.item(), total_loss.item()))
        optimizer.step()
        lr_scheduler.step()
        train_step += 1
    summary_writer.flush()


def eval(epoch, model, data_loader, loss_fn, summary_writer):
    model.eval()
    with torch.no_grad():
        all_duration_loss = []
        all_pitch_loss = []
        all_energy_loss = []
        all_mel_loss = []
        all_postnet_mel_loss = []
        for i, (keys, speakers, durations, text, mel, pitch, energy,
                text_length, mel_length, token_types,
                _) in enumerate(data_loader):
            speakers = speakers.cuda()
            durations = durations.cuda()
            text = text.cuda()
            mel = mel.cuda()
            pitch = pitch.cuda()
            energy = energy.cuda()
            text_length = text_length.cuda()
            mel_length = mel_length.cuda()
            token_types = token_types.cuda()
            (mel_prediction, postnet_mel_prediction, mel_mask, mel_len,
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

            all_duration_loss.append(duration_loss)
            all_pitch_loss.append(pitch_loss)
            all_energy_loss.append(energy_loss)
            all_mel_loss.append(mel_loss)
            all_postnet_mel_loss.append(postnet_mel_loss)
            # randomly select one audio from the first eval batch to plot
            if i == 0:
                mel_idx = random.randrange(0, mel.shape[0])
                mels_to_plot = [
                    mel[mel_idx].cpu(), mel_prediction[mel_idx].cpu(),
                    postnet_mel_prediction[mel_idx].cpu()
                ]

        def get_average_loss(loss_list):
            return sum(loss_list) / len(loss_list)

        avg_dur_loss = get_average_loss(all_duration_loss)
        avg_pitch_loss = get_average_loss(all_pitch_loss)
        avg_energy_loss = get_average_loss(all_energy_loss)
        avg_mel_loss = get_average_loss(all_mel_loss)
        avg_postnet_mel_loss = get_average_loss(all_postnet_mel_loss)
        avg_total_loss = (avg_dur_loss + avg_pitch_loss + avg_energy_loss +
                          avg_mel_loss + avg_postnet_mel_loss)
        summary_writer.add_scalar('average eval duration loss',
                                  avg_dur_loss.item(), epoch)
        summary_writer.add_scalar('average eval pitch loss',
                                  avg_pitch_loss.item(), epoch)
        summary_writer.add_scalar('average eval energy loss',
                                  avg_energy_loss.item(), epoch)
        summary_writer.add_scalar('average eval mel loss', avg_mel_loss.item(),
                                  epoch)
        summary_writer.add_scalar('average eval postnet mel loss',
                                  avg_postnet_mel_loss.item(), epoch)
        summary_writer.add_scalar('average eval total loss',
                                  avg_total_loss.item(), epoch)
        print(("epoch {}, eval duration loss {}, "
               "avg eval pitch loss {}, avg eval energy loss {}, "
               "avg eval mel loss {}, "
               "avg eval postnet mel loss {}, avg eval total loss: {}").format(
                   epoch, avg_dur_loss.item(), avg_pitch_loss.item(),
                   avg_energy_loss.item(), avg_mel_loss.item(),
                   avg_postnet_mel_loss.item(), avg_total_loss.item()))

        summary_writer.add_figure(
            'eval mels',
            plot_mel(mels_to_plot,
                     ['ground truth', 'prediction', 'postnet prediction']),
            epoch)
    summary_writer.flush()


def save_ckpt(path, model, lr_scheduler, optimizer, train_step, epoch):
    torch.save([
        model.state_dict(),
        lr_scheduler.state_dict(),
        optimizer.state_dict(), train_step, epoch
    ], path)


def load_ckpt(path):
    with open(path, 'rb') as fin:
        (model_state_dict, lr_scheduler_state_dict, optimizer_state_dict,
         train_step, epoch) = torch.load(fin, 'cpu')
        return (model_state_dict, lr_scheduler_state_dict,
                optimizer_state_dict, train_step, epoch)


def main(args):
    global train_step
    with open(args.config, 'r') as fin:
        conf = config.load_cfg(fin)

    log_dir = pathlib.Path(args.log_dir)
    tensorboard_dir = log_dir / 'tensorboard'
    checkpoint_dir = log_dir / 'ckpt'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    (train_dataset, _, pitch_stats, energy_stats, phn2id,
     spk2id) = FastSpeech2TrainingDataset(args.train_data_list,
                                          args.batch_size, args.spk2id_file,
                                          args.phn2id_file,
                                          args.special_tokens_file,
                                          args.cmvn_dir, conf)
    val_dataset, *_ = FastSpeech2TrainingDataset(
        args.val_data_list, args.batch_size, args.spk2id_file,
        args.phn2id_file, args.special_tokens_file, args.cmvn_dir, conf)
    pitch_mean, pitch_sigma, pitch_min, pitch_max = pitch_stats
    energy_mean, energy_sigma, energy_min, energy_max = energy_stats

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   num_workers=args.num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)

    model = FastSpeech2(
        conf.model.d_model, conf.model.n_enc_layer, conf.model.n_enc_head,
        conf.model.n_enc_conv_filter, conf.model.enc_conv_kernel_size,
        conf.model.enc_dropout, len(phn2id), conf.model.padding_idx,
        conf.model.n_va_conv_filter, conf.model.va_conv_kernel_size,
        conf.model.va_dropout, pitch_min, pitch_max, pitch_mean, pitch_sigma,
        energy_min, energy_max, energy_mean, energy_sigma,
        conf.model.n_pitch_bin, conf.model.n_energy_bin,
        conf.model.n_dec_layer, conf.model.n_dec_head,
        conf.model.n_dec_conv_filter,
        conf.model.dec_conv_kernel_size, conf.model.dec_dropout, conf.n_mels,
        len(spk2id), conf.model.postnet_kernel_size,
        conf.model.postnet_hidden_dim, conf.model.n_postnet_conv_layers,
        conf.model.postnet_dropout, conf.model.max_pos_enc_len)
    model = model.cuda()

    loss_fn = loss.FastSpeech2Loss()

    optimizer = Adam(model.parameters(), 1, conf.optimizer.betas,
                     conf.optimizer.eps)
    lr_scheduler = transformer_lr_scheduler(optimizer, conf.model.d_model,
                                            conf.optimizer.warmup_steps)
    writer = SummaryWriter(tensorboard_dir)
    last_epoch = 0
    if args.ckpt:
        (model_state_dict, lr_scheduler_state_dict, optimizer_state_dict,
         train_step, last_epoch) = load_ckpt(args.ckpt)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        print("Resume training from epoch {}.".format(last_epoch))

    for epoch in range(last_epoch + 1, args.epoch + 1):
        train(epoch, model, train_data_loader, loss_fn, optimizer,
              lr_scheduler, writer)
        eval(epoch, model, val_data_loader, loss_fn, writer)
        save_ckpt(checkpoint_dir / 'fastspeech2_{}.ckpt'.format(epoch), model,
                  lr_scheduler, optimizer, train_step, epoch)


if __name__ == '__main__':
    main(get_args())
