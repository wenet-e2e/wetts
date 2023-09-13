# Copyright (c) 2022, Yongqiang Li (yongqiangli@alumni.hust.edu.cn)
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
import time

import numpy as np
from scipy.io import wavfile
import torch

from models import SynthesizerTrn
import utils


def get_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--checkpoint", required=True, help="checkpoint")
    parser.add_argument("--cfg", required=True, help="config file")
    parser.add_argument("--outdir", required=True, help="ouput directory")
    parser.add_argument("--phone_table", required=True, help="input phone dict")
    parser.add_argument("--speaker_table", default=True, help="speaker table")
    parser.add_argument("--test_file", required=True, help="test file")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id for this local rank, -1 for cpu"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    phone_dict = {}
    for line in open(args.phone_table):
        phone_id = line.strip().split()
        phone_dict[phone_id[0]] = int(phone_id[1])

    speaker_dict = {}
    for line in open(args.speaker_table):
        arr = line.strip().split()
        assert len(arr) == 2
        speaker_dict[arr[0]] = int(arr[1])
    hps = utils.get_hparams_from_file(args.cfg)

    net_g = SynthesizerTrn(
        len(phone_dict),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=len(speaker_dict),
        **hps.model
    )
    net_g = net_g.to(device)

    net_g.eval()
    utils.load_checkpoint(args.checkpoint, net_g, None)

    for line in open(args.test_file):
        audio_path, speaker, text = line.strip().split("|")
        sid = speaker_dict[speaker]
        seq = [phone_dict[symbol] for symbol in text.split()]
        seq = torch.LongTensor(seq)
        print(audio_path)
        with torch.no_grad():
            x = seq.to(device).unsqueeze(0)
            x_length = torch.LongTensor([seq.size(0)]).to(device)
            sid = torch.LongTensor([sid]).to(device)
            st = time.time()
            audio = (
                net_g.infer(
                    x,
                    x_length,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1,
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
            audio *= 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
            print(
                "RTF {}".format(
                    (time.time() - st) / (audio.shape[0] / hps.data.sampling_rate)
                )
            )
            sys.stdout.flush()
            audio = np.clip(audio, -32767.0, 32767.0)
            wavfile.write(
                args.outdir + "/" + audio_path.split("/")[-1],
                hps.data.sampling_rate,
                audio.astype(np.int16),
            )


if __name__ == "__main__":
    main()
