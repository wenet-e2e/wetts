#!/usr/bin/env python
# Copyright 2022 Binbin Zhang(binbzha@qq.com)

import os
import sys

train_dir = sys.argv[1]
out_dir = sys.argv[2]

content_path = os.path.join(train_dir, 'content.txt')
wav_scp = os.path.join(out_dir, 'wav.scp')
label_scp = os.path.join(out_dir, 'text')

with open(content_path, 'r', encoding='utf8') as fin, \
     open(wav_scp, 'w', encoding='utf8') as fout_wav, \
     open(label_scp, 'w', encoding='utf8') as fout_lab:
    for line in fin:
        arr = line.strip().split()
        key = arr[0][:-4]
        speaker = key[:-4]
        lab = ' '.join(arr[2::2])
        label_path = os.path.join(out_dir, 'lab', speaker,
                                  '{}.lab'.format(key))
        dir_name = os.path.dirname(label_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        wav_path = os.path.join(train_dir, 'wav', speaker,
                                '{}.wav'.format(key))
        fout_wav.write('{} {}\n'.format(key, wav_path))
        fout_lab.write('{} {}\n'.format(key, lab))
        with open(label_path, 'w', encoding='utf8') as fout:
            fout.write(lab)
