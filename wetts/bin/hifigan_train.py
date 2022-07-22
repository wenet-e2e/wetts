# Copyright (c) 2022 Tsinghua University(Jie Chen)
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
# Modified from HiFi-GAN(https://github.com/jik876/hifi-gan)

import argparse
import itertools
import pathlib
import random

import numpy as np
import jsonlines
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yacs.config
import scipy.io.wavfile

from wetts.utils import file_utils
from wetts.models.am.fastspeech2.fastspeech2 import FastSpeech2
from wetts.models.am.fastspeech2.module.dataset import (
    FastSpeech2TrainingDataset)
from wetts.models.vocoder.hifigan.hifigan import (Generator,
                                                  MultiPeriodDiscriminator,
                                                  MultiScaleDiscriminator)
from wetts.models.vocoder.hifigan.module import loss
from wetts.models.vocoder.hifigan.module.dataset import (
    HiFiGANFinetuneDataset, HiFiGANTrainingDataset)


def get_args(argv=None):
    parser = argparse.ArgumentParser()

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--num_workers',
                               type=int,
                               default=2,
                               required=True,
                               help='number of workers for dataloader')
    common_parser.add_argument('--batch_size_hifigan',
                               type=int,
                               default=32,
                               required=True,
                               help='batch size for hifigan finetuning')
    common_parser.add_argument(
        '--fastspeech2_train_datalist',
        type=str,
        required=True,
        help='path to fastspeech2 training data list file')
    common_parser.add_argument('--hifigan_config',
                               required=True,
                               help='path to hifigan config file')
    common_parser.add_argument(
        '--hifigan_ckpt',
        nargs=2,
        help='path to hifigan generator and discriminator checkpoint '
        'for resume training or finetuning hifigan')
    common_parser.add_argument(
        '--epoch',
        default=1000,
        type=int,
        help='number of epochs for training or finetuning hifigan')
    common_parser.add_argument(
        '--export_dir',
        type=str,
        required=True,
        help='path to directory for exporting finetuned HiFiGAN')

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', parents=[common_parser])
    finetune_parser = subparsers.add_parser('finetune',
                                            parents=[common_parser])

    finetune_parser.add_argument('--batch_size_fastspeech2',
                                 type=int,
                                 default=32,
                                 required=True,
                                 help='batch size for fastspeech2 inference')
    finetune_parser.add_argument('--fastspeech2_config',
                                 type=str,
                                 required=True,
                                 help='path to fastspeech2 config file')
    finetune_parser.add_argument('--phn2id_file',
                                 type=str,
                                 required=True,
                                 help='path to phn2id file')
    finetune_parser.add_argument(
        '--spk2id_file',
        type=str,
        required=True,
        help='path to spk2id file, this file must be provided '
        'for both multi-speaker FastSpeech2 and single-speaker '
        'FastSpeech2')
    finetune_parser.add_argument('--special_tokens_file',
                                 type=str,
                                 required=True,
                                 help='path to special tokens file')
    finetune_parser.add_argument('--cmvn_dir',
                                 type=str,
                                 required=True,
                                 help='path to cmvn dir')
    finetune_parser.add_argument(
        '--fastspeech2_ckpt',
        type=str,
        required=True,
        help='path to fastspeech2 ckpt to generate mel')
    finetune_parser.add_argument(
        '--generate_samples',
        action='store_true',
        help='Whether generate samples for finetuning. '
        'Should not be set when resume fintuning.')

    train_parser.add_argument(
        '--fastspeech2_val_datalist',
        type=str,
        required=True,
        help='path to fastspeech2 validation data list file')

    train_parser.set_defaults(func=load_train_dataset)
    finetune_parser.set_defaults(func=load_finetune_dataset)

    args = parser.parse_args(argv)
    return args


def export_fastspeech2_mel_wav(fastspeech2_data_list, spk2id_file, phn2id_file,
                               special_tokens_file, cmvn_dir, fastspeech2_conf,
                               export_dir, num_workers, fastspeech2_ckpt,
                               batch_size):
    export_dir.mkdir(parents=True, exist_ok=True)
    (fastspeech2_dataset, mel_stats, pitch_stats, energy_stats, phn2id,
     spk2id) = FastSpeech2TrainingDataset(fastspeech2_data_list, batch_size,
                                          spk2id_file, phn2id_file,
                                          special_tokens_file, cmvn_dir,
                                          fastspeech2_conf)
    pitch_mean, pitch_sigma, pitch_min, pitch_max = pitch_stats
    energy_mean, energy_sigma, energy_min, energy_max = energy_stats
    mel_mean, mel_sigma = mel_stats
    fastspeech2_data_loader = DataLoader(fastspeech2_dataset,
                                         batch_size=None,
                                         num_workers=num_workers)
    fastspeech2 = FastSpeech2(
        fastspeech2_conf.model.d_model, fastspeech2_conf.model.n_enc_layer,
        fastspeech2_conf.model.n_enc_head,
        fastspeech2_conf.model.n_enc_conv_filter,
        fastspeech2_conf.model.enc_conv_kernel_size,
        fastspeech2_conf.model.enc_dropout, len(phn2id),
        fastspeech2_conf.model.padding_idx,
        fastspeech2_conf.model.n_va_conv_filter,
        fastspeech2_conf.model.va_conv_kernel_size,
        fastspeech2_conf.model.va_dropout, pitch_min, pitch_max, pitch_mean,
        pitch_sigma, energy_min, energy_max, energy_mean, energy_sigma,
        fastspeech2_conf.model.n_pitch_bin,
        fastspeech2_conf.model.n_energy_bin,
        fastspeech2_conf.model.n_dec_layer, fastspeech2_conf.model.n_dec_head,
        fastspeech2_conf.model.n_dec_conv_filter,
        fastspeech2_conf.model.dec_conv_kernel_size,
        fastspeech2_conf.model.dec_dropout, fastspeech2_conf.n_mels,
        len(spk2id), fastspeech2_conf.model.postnet_kernel_size,
        fastspeech2_conf.model.postnet_hidden_dim,
        fastspeech2_conf.model.n_postnet_conv_layers,
        fastspeech2_conf.model.postnet_dropout,
        fastspeech2_conf.model.max_pos_enc_len)
    (fastspeech2_state_dict, _, _, fastspeech2_steps,
     *_) = file_utils.load_ckpt(fastspeech2_ckpt)
    print("Loading fastspeech2 from step {}.".format(fastspeech2_steps))

    fastspeech2.load_state_dict(fastspeech2_state_dict)
    fastspeech2 = fastspeech2.cuda()
    fastspeech2.eval()
    print(
        "Generating mel-spectrogram from FastSpeech2 for HiFiGAN finetuning.")
    file_list = []
    with torch.no_grad():
        for (keys, speakers, durations, text, mel, _, _, text_length, _,
             token_types, wav) in fastspeech2_data_loader:
            speakers = speakers.cuda()
            durations = durations.cuda()
            text = text.cuda()
            mel = mel.cuda()
            text_length = text_length.cuda()
            token_types = token_types.cuda()
            # Only provide ground truth duration here to ensure FastSpeech2 can
            # generate mel which has the same length as ground truth.
            (_, postnet_mel_prediction, mel_mask,
             *_) = fastspeech2(text,
                               text_length,
                               token_types,
                               duration_target=durations,
                               speaker=speakers)
            for name, mel_prediction, wav_target, mel_len in zip(
                    keys, postnet_mel_prediction, wav, (~mel_mask).sum(dim=1)):
                # mel_prediction: (t,d)
                # wav_target: (t)
                mel_prediction_filepath = (
                    export_dir /
                    '{}_mel_prediction.npy'.format(name)).resolve()
                wav_target_filepath = (
                    export_dir / '{}_wav_target.wav'.format(name)).resolve()
                np.save(
                    mel_prediction_filepath,
                    mel_prediction.cpu().numpy()[:mel_len] * mel_sigma +
                    mel_mean)

                # Save wavs processed by fastspeech2 dataloader in fastspeech2's
                # sample rate. If you use a different sample rate from HiFiGAN,
                # a resampling operation may be needed.
                scipy.io.wavfile.write(wav_target_filepath,
                                       fastspeech2_conf.sr,
                                       wav_target.cpu().numpy())
                file_list.append({
                    'mel_prediction_filepath':
                    str(mel_prediction_filepath),
                    'wav_target_filepath':
                    str(wav_target_filepath)
                })
    with jsonlines.open(export_dir / 'hifigan_finetune_data_list.jsonl',
                        'w') as f:
        f.write_all(file_list)
    print("Finished mel-spectrogram generating!")


def train(hifigan_conf,
          hifigan_ckpt,
          export_dir,
          num_workers,
          total_epoch,
          hifigan_train_dataset,
          hifigan_val_dataset=None):
    export_dir = pathlib.Path(export_dir)
    hifigan_generator = Generator(hifigan_conf.model.resblock_kernel_sizes,
                                  hifigan_conf.model.resblock_dilation_sizes,
                                  hifigan_conf.model.upsample_rates,
                                  hifigan_conf.model.upsample_kernel_sizes,
                                  hifigan_conf.model.upsample_initial_channel,
                                  hifigan_conf.model.resblock_type)
    hifigan_mpd = MultiPeriodDiscriminator()
    hifigan_msd = MultiScaleDiscriminator()

    if hifigan_ckpt:
        hifigan_generator_ckpt, hifigan_discriminator_ckpt = hifigan_ckpt
        hifigan_generator_state_dict = file_utils.load_ckpt(
            hifigan_generator_ckpt)
        hifigan_discriminator_state_dict = file_utils.load_ckpt(
            hifigan_discriminator_ckpt)

        print("loading HiFiGAN from step {}.".format(
            hifigan_discriminator_state_dict['steps']))
        hifigan_generator.load_state_dict(
            hifigan_generator_state_dict['generator'])
        hifigan_mpd.load_state_dict(hifigan_discriminator_state_dict['mpd'])
        hifigan_msd.load_state_dict(hifigan_discriminator_state_dict['msd'])

    hifigan_generator = hifigan_generator.cuda()
    hifigan_mpd = hifigan_mpd.cuda()
    hifigan_msd = hifigan_msd.cuda()

    optim_g = optim.AdamW(hifigan_generator.parameters(),
                          hifigan_conf.optimizer.lr,
                          hifigan_conf.optimizer.betas)
    optim_d = optim.AdamW(
        itertools.chain(hifigan_msd.parameters(), hifigan_mpd.parameters()),
        hifigan_conf.optimizer.lr, hifigan_conf.optimizer.betas)
    start_epoch = 1
    step = 1
    if hifigan_ckpt:
        optim_g.load_state_dict(hifigan_discriminator_state_dict['optim_g'])
        optim_d.load_state_dict(hifigan_discriminator_state_dict['optim_d'])
        start_epoch = hifigan_discriminator_state_dict['epoch'] + 1
        step = hifigan_discriminator_state_dict['steps'] + 1
        scheduler_g = lr_scheduler.ExponentialLR(
            optim_g, hifigan_conf.optimizer.lr_decay,
            hifigan_discriminator_state_dict['epoch'])
        scheduler_d = lr_scheduler.ExponentialLR(
            optim_d, hifigan_conf.optimizer.lr_decay,
            hifigan_discriminator_state_dict['epoch'])

    else:
        scheduler_g = lr_scheduler.ExponentialLR(
            optim_g, hifigan_conf.optimizer.lr_decay, -1)
        scheduler_d = lr_scheduler.ExponentialLR(
            optim_d, hifigan_conf.optimizer.lr_decay, -1)
    hifigan_train_dataloader = DataLoader(hifigan_train_dataset,
                                          batch_size=None,
                                          num_workers=num_workers)
    if hifigan_val_dataset is not None:
        hifigan_val_dataloader = DataLoader(hifigan_val_dataset,
                                            batch_size=None,
                                            num_workers=num_workers)
    else:
        hifigan_val_dataloader = None
    hifigan_generator.train()
    hifigan_mpd.train()
    hifigan_msd.train()
    hifigan_mel_loss = loss.HiFiGANMelLoss(hifigan_conf.sr, hifigan_conf.n_fft,
                                           hifigan_conf.n_mels,
                                           hifigan_conf.hop_length,
                                           hifigan_conf.win_length,
                                           hifigan_conf.fmin).cuda()
    writer = SummaryWriter(export_dir / 'log')

    for i in range(start_epoch, start_epoch + total_epoch):
        for (mel_clip, wav_clip, _, _) in hifigan_train_dataloader:
            mel_clip = mel_clip.permute(0, 2, 1).cuda()
            wav_clip = wav_clip.cuda()

            gen_wav_clip = hifigan_generator(mel_clip)
            # gen_wav_clip: b,1,t

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = hifigan_mpd(wav_clip.unsqueeze(1),
                                                       gen_wav_clip.detach())
            (loss_disc_f, losses_disc_f_r,
             losses_disc_f_g) = loss.discriminator_loss(
                 y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = hifigan_msd(wav_clip.unsqueeze(1),
                                                       gen_wav_clip.detach())
            (loss_disc_s, losses_disc_s_r,
             losses_disc_s_g) = loss.discriminator_loss(
                 y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = hifigan_mel_loss(gen_wav_clip.squeeze(1), wav_clip)

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = hifigan_mpd(
                wav_clip.unsqueeze(1), gen_wav_clip)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = hifigan_msd(
                wav_clip.unsqueeze(1), gen_wav_clip)
            loss_fm_f = loss.feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = loss.feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = loss.generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = loss.generator_loss(y_ds_hat_g)
            loss_gen_all = (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f +
                            loss_mel * 45)

            loss_gen_all.backward()
            optim_g.step()
            print('epoch: {}, step: {}, total generator loss: {}, '
                  'total discriminator loss: {}, mpd loss: {}, msd loss: {}, '
                  'loss_fm_f: {}, loss_fm_s: {}, loss_gen_f: {}, '
                  'loss_gen_s: {}, loss_mel: {}'.format(
                      i, step, loss_gen_all.item(), loss_disc_all.item(),
                      loss_disc_f.item(), loss_disc_s.item(), loss_fm_f.item(),
                      loss_fm_s.item(), loss_gen_f.item(), loss_gen_s.item(),
                      loss_mel.item()))
            writer.add_scalar('total generator loss', loss_gen_all.item(),
                              step)
            writer.add_scalar('total discriminator loss', loss_disc_all.item(),
                              step)
            writer.add_scalar('MPD loss', loss_disc_f.item(), step)
            writer.add_scalar('MSD loss', loss_disc_s.item(), step)
            writer.add_scalar('loss_fm_f', loss_fm_f.item(), step)
            writer.add_scalar('loss_fm_s', loss_fm_s.item(), step)
            writer.add_scalar('loss_gen_f', loss_gen_f.item(), step)
            writer.add_scalar('loss_gen_s', loss_gen_s.item(), step)
            writer.add_scalar('loss_mel', loss_mel.item(), step)
            step += 1
        scheduler_g.step()
        scheduler_d.step()

        with torch.no_grad():
            if hifigan_val_dataloader is not None:
                total_loss = []
                hifigan_generator.eval()
                for mel_clip, wav_clip, mels, wavs in hifigan_val_dataloader:
                    mel_clip = mel_clip.permute(0, 2, 1).cuda()
                    wav_clip = wav_clip.cuda()
                    gen_wav_clip = hifigan_generator(mel_clip)
                    loss_mel = hifigan_mel_loss(gen_wav_clip.squeeze(1),
                                                wav_clip)
                    total_loss.append(loss_mel.item())

                # randomly pick up one full mel from last batch
                # to get reconstructed speech
                random_idx = random.randrange(len(mels))
                random_wav = wavs[random_idx]
                random_mel = mels[random_idx]
                gen_wav = hifigan_generator(
                    random_mel.unsqueeze(0).permute(0, 2, 1).cuda()).squeeze(1)

                mean_loss = np.mean(np.array(total_loss))
                print('epoch: {}, mean valiation mel loss: {}'.format(
                    i, mean_loss))
                writer.add_scalar('mean validation mel loss', mean_loss, i)
                writer.add_audio('validation predicted audio',
                                 gen_wav,
                                 i,
                                 sample_rate=hifigan_conf.sr)
                writer.add_audio('validation ground truth audio',
                                 random_wav,
                                 i,
                                 sample_rate=hifigan_conf.sr)
            # saving ckpt in original HiFiGAN format for compatibility
            torch.save({'generator': hifigan_generator.state_dict()},
                       export_dir / 'g_{}'.format(i))
            torch.save(
                {
                    'mpd': hifigan_mpd.state_dict(),
                    'msd': hifigan_msd.state_dict(),
                    'steps': step,
                    'epoch': i,
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                }, export_dir / 'd_{}'.format(i))
        writer.flush()


def main(args):
    """Running HiFiGAN training or finetuning.

    `args.func` will be determined by command line argument.
    For `python hifigan_train.py train ...`, it's `load_train_dataset`.
    For `python hifigan_train.py finetune ...`, it's `load_train_dataset`
    """
    hifigan_conf, hifigan_train_dataset, hifigan_val_dataset = args.func(args)
    train(hifigan_conf, args.hifigan_ckpt, args.export_dir, args.num_workers,
          args.epoch, hifigan_train_dataset, hifigan_val_dataset)


def load_finetune_dataset(args):
    with open(args.hifigan_config) as f1, open(args.fastspeech2_config) as f2:
        hifigan_conf = yacs.config.load_cfg(f1)
        fastspeech2_conf = yacs.config.load_cfg(f2)
    export_dir = pathlib.Path(args.export_dir)
    if args.generate_samples:
        export_fastspeech2_mel_wav(args.fastspeech2_train_datalist,
                                   args.spk2id_file, args.phn2id_file,
                                   args.special_tokens_file, args.cmvn_dir,
                                   fastspeech2_conf,
                                   export_dir / 'vocoder_finetune_dataset',
                                   args.num_workers, args.fastspeech2_ckpt,
                                   args.batch_size_fastspeech2)
    hifigan_train_dataset = HiFiGANFinetuneDataset(
        export_dir / 'vocoder_finetune_dataset' /
        'hifigan_finetune_data_list.jsonl', hifigan_conf,
        args.batch_size_hifigan)
    hifigan_val_dataset = None
    return hifigan_conf, hifigan_train_dataset, hifigan_val_dataset


def load_train_dataset(args):
    with open(args.hifigan_config) as f2:
        hifigan_conf = yacs.config.load_cfg(f2)
    hifigan_train_dataset = HiFiGANTrainingDataset(
        args.fastspeech2_train_datalist, hifigan_conf, args.batch_size_hifigan)
    hifigan_val_dataset = HiFiGANTrainingDataset(args.fastspeech2_val_datalist,
                                                 hifigan_conf,
                                                 args.batch_size_hifigan)
    return hifigan_conf, hifigan_train_dataset, hifigan_val_dataset


if __name__ == '__main__':
    main(get_args())
