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
import pathlib

import scipy.io.wavfile
import torch
from torch.utils.data import DataLoader
import yacs.config
import tqdm

from wetts.models.vocoder.hifigan.hifigan import Generator
from wetts.models.vocoder.hifigan.module.dataset import HiFiGANInferenceDataset
from wetts.utils import file_utils


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',
                        type=int,
                        default=2,
                        help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--config',
                        required=True,
                        type=str,
                        help='hifigan config file')
    parser.add_argument('--datalist',
                        required=True,
                        type=str,
                        help='path to datalist')
    parser.add_argument('--ckpt',
                        required=True,
                        help='path to hifigan generator checkpoint')
    parser.add_argument(
        '--export_dir',
        required=True,
        help='path to directory for exporting inference results')
    parser.add_argument('--max_wav_value',
                        default=32768,
                        type=int,
                        help='maximum wav value to recover sample points')
    args = parser.parse_args(argv)
    return args


def inference(hifigan_conf, hifigan_ckpt, export_dir, batch_size, num_workers,
              datalist, max_wav_value):
    hifigan_generator = Generator(hifigan_conf.model.resblock_kernel_sizes,
                                  hifigan_conf.model.resblock_dilation_sizes,
                                  hifigan_conf.model.upsample_rates,
                                  hifigan_conf.model.upsample_kernel_sizes,
                                  hifigan_conf.model.upsample_initial_channel,
                                  hifigan_conf.model.resblock_type)
    hifigan_state_dict = file_utils.load_ckpt(hifigan_ckpt)
    print('loading HiFiGAN checkpoint: {}'.format(hifigan_ckpt))
    hifigan_generator.load_state_dict(hifigan_state_dict['generator'])
    hifigan_inference_dataset = HiFiGANInferenceDataset(datalist, batch_size)
    hifigan_inference_dataloader = DataLoader(hifigan_inference_dataset,
                                              batch_size=None,
                                              num_workers=num_workers)
    hifigan_generator = hifigan_generator.cuda()
    hifigan_generator.eval()
    hifigan_generator.remove_weight_norm()
    with torch.no_grad():
        output = []
        for names, mels, lengths in tqdm.tqdm(hifigan_inference_dataloader):
            # mels: b,t,d -> b,d,t
            mels = mels.permute(0, 2, 1).cuda()
            # wav_prediction: b,t
            wav_prediction = hifigan_generator(mels).squeeze(1)
            for name, wav, l in zip(names, wav_prediction, lengths):
                output.append(
                    (name, wav[:l * hifigan_conf.hop_length] * max_wav_value))
        for name, wav in output:
            scipy.io.wavfile.write(export_dir / '{}.wav'.format(name),
                                   hifigan_conf.sr,
                                   wav.cpu().numpy().astype('int16'))


def main(args):
    with open(args.config) as f:
        hifigan_conf = yacs.config.load_cfg(f)
    export_dir = pathlib.Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    inference(hifigan_conf, args.ckpt, export_dir, args.batch_size,
              args.num_workers, args.datalist, args.max_wav_value)


if __name__ == '__main__':
    main(get_args())
