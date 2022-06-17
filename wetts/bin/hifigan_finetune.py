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
from wetts.models.vocoder.hifigan.module.dataset import HiFiGANFinetuneDataset


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='number of workers for dataloader')
    parser.add_argument('--batch_size_hifigan',
                        type=int,
                        default=32,
                        help='batch size for hifigan finetuning')
    parser.add_argument('--fastspeech2_config',
                        required=True,
                        help='fastspeech2 config file')
    parser.add_argument('--fastspeech2_datalist',
                        required=True,
                        help='fastspeech2 data list file')
    parser.add_argument('--hifigan_config',
                        required=True,
                        help='hifigan config file')
    parser.add_argument('--phn2id_file', required=True, help='phn2id file')
    parser.add_argument('--spk2id_file', required=True, help='spk2id file')
    parser.add_argument('--special_tokens_file',
                        required=True,
                        help='special tokens file')
    parser.add_argument('--cmvn_dir', required=True, help='cmvn dir')
    parser.add_argument('--fastspeech2_ckpt',
                        required=True,
                        help='path to fastspeech2 ckpt to generate mel')
    parser.add_argument('--hifigan_ckpt',
                        required=True,
                        nargs=2,
                        help='path to hifigan ckpt to finetune')
    parser.add_argument('--finetune_epoch',
                        default=1000,
                        type=int,
                        help='number of epochs for finetuning hifigan')
    parser.add_argument(
        '--export_dir',
        required=True,
        help='path to directory for exporting finetuned HiFiGAN')
    args = parser.parse_args(argv)
    return args


def export_fastspeech2_mel_wav(fastspeech2_data_list, spk2id_file, phn2id_file,
                               special_tokens_file, cmvn_dir, fastspeech2_conf,
                               export_dir, num_workers, fastspeech2_ckpt):
    export_dir.mkdir(parents=True, exist_ok=True)
    (fastspeech2_dataset, mel_stats, pitch_stats, energy_stats,
     phn2id) = FastSpeech2TrainingDataset(fastspeech2_data_list, spk2id_file,
                                          phn2id_file, special_tokens_file,
                                          cmvn_dir, fastspeech2_conf)
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
        fastspeech2_conf.n_speaker, fastspeech2_conf.model.postnet_kernel_size,
        fastspeech2_conf.model.postnet_hidden_dim,
        fastspeech2_conf.model.n_postnet_conv_layers,
        fastspeech2_conf.model.postnet_dropout,
        fastspeech2_conf.model.max_pos_enc_len)
    fastspeech2_state_dict, _, _, fastspeech2_steps, _, _ = file_utils.load_ckpt(
        fastspeech2_ckpt)
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
                        mode='w') as f:
        f.write_all(file_list)
    print("Finished mel-spectrogram generating!")


def finetune(hifigan_conf, hifigan_ckpt, export_dir, batch_size, num_workers,
             finetune_epoch):
    hifigan_generator = Generator(hifigan_conf.model.resblock_kernel_sizes,
                                  hifigan_conf.model.resblock_dilation_sizes,
                                  hifigan_conf.model.upsample_rates,
                                  hifigan_conf.model.upsample_kernel_sizes,
                                  hifigan_conf.model.upsample_initial_channel,
                                  hifigan_conf.model.resblock_type)
    hifigan_mpd = MultiPeriodDiscriminator()
    hifigan_msd = MultiScaleDiscriminator()

    hifigan_generator_ckpt, hifigan_discriminator_ckpt = hifigan_ckpt
    hifigan_generator_state_dict = file_utils.load_ckpt(hifigan_generator_ckpt)
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

    optim_g.load_state_dict(hifigan_discriminator_state_dict['optim_g'])
    optim_d.load_state_dict(hifigan_discriminator_state_dict['optim_d'])

    scheduler_g = lr_scheduler.ExponentialLR(
        optim_g, hifigan_conf.optimizer.lr_decay,
        hifigan_discriminator_state_dict['epoch'])
    scheduler_d = lr_scheduler.ExponentialLR(
        optim_d, hifigan_conf.optimizer.lr_decay,
        hifigan_discriminator_state_dict['epoch'])

    hifigan_dataset = HiFiGANFinetuneDataset(
        export_dir / 'vocoder_finetune_dataset' /
        'hifigan_finetune_data_list.jsonl', hifigan_conf.segment_size,
        hifigan_conf.hop_length, batch_size)
    hifigan_dataloader = DataLoader(hifigan_dataset,
                                    batch_size=None,
                                    num_workers=num_workers)
    hifigan_generator.train()
    hifigan_mpd.train()
    hifigan_msd.train()
    hifigan_mel_loss = loss.HiFiGANMelLoss(hifigan_conf.sr, hifigan_conf.n_fft,
                                           hifigan_conf.n_mels,
                                           hifigan_conf.hop_length,
                                           hifigan_conf.win_length,
                                           hifigan_conf.fmin, None).cuda()
    writer = SummaryWriter(export_dir / 'finetune_log')
    print('Start HiFiGAN finetuning!')
    epoch = hifigan_discriminator_state_dict['epoch'] + 1
    step = hifigan_discriminator_state_dict['steps'] + 1
    for i in range(epoch, finetune_epoch + epoch):
        for (mel_prediction_clip, wav_clip) in hifigan_dataloader:
            mel_prediction_clip = mel_prediction_clip.permute(0, 2, 1).cuda()
            wav_clip = wav_clip.cuda()

            gen_wav_clip = hifigan_generator(mel_prediction_clip)
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
                            loss_mel)

            loss_gen_all.backward()
            optim_g.step()
            print('epoch: {}, step: {}, mpd loss: {}, msd loss: {}, '
                  'loss_fm_f: {}, loss_fm_s: {}, loss_gen_f: {}, '
                  'loss_gen_s: {}, loss_mel: {}'.format(
                      i + 1, step, loss_disc_f.item(), loss_disc_s.item(),
                      loss_fm_f.item(), loss_fm_s.item(), loss_gen_f.item(),
                      loss_gen_s.item(), loss_mel.item()))
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
            export_dir = pathlib.Path(export_dir)
            # saving ckpt in original HiFiGAN format for compatibility
            torch.save({'generator': hifigan_generator.state_dict()},
                       export_dir / 'g_finetune_{}'.format(i))
            torch.save({
                'mpd': hifigan_mpd.state_dict(),
                'msd': hifigan_msd.state_dict(),
                'steps': step,
                'epoch': i,
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
            })
        writer.flush()


def main(args):
    with open(args.fastspeech2_config) as f1, open(args.hifigan_config) as f2:
        fastspeech2_conf = yacs.config.load_cfg(f1)
        hifigan_conf = yacs.config.load_cfg(f2)
    export_dir = pathlib.Path(args.export_dir)
    export_fastspeech2_mel_wav(args.fastspeech2_datalist, args.spk2id_file,
                               args.phn2id_file, args.special_tokens_file,
                               args.cmvn_dir, fastspeech2_conf,
                               export_dir / 'vocoder_finetune_dataset',
                               args.num_workers, args.fastspeech2_ckpt)
    finetune(hifigan_conf, args.hifigan_ckpt, export_dir,
             args.batch_size_hifigan, args.num_workers, args.finetune_epoch)


if __name__ == '__main__':
    main(get_args())
