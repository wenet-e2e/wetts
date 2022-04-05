#!/usr/bin/env python3

# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com)
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
import io
import json
import logging
import os
import random
import tarfile
import time
import multiprocessing

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def write_tar_file(data_list, tar_file, index=0, total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    save_time = 0.0
    write_time = 0.0
    with tarfile.open(tar_file, 'w') as tar:
        for item in data_list:
            key, wav, dur, phones, speaker = item
            suffix = wav.split('.')[-1]
            assert suffix in AUDIO_FORMAT_SETS
            ts = time.time()
            with open(wav, 'rb') as fin:
                data = fin.read()
            read_time += (time.time() - ts)
            ts = time.time()

            json_file = key + '.json'
            obj = {'duration': dur, 'phones': phones, 'speaker': speaker}
            json_data = json.dumps(obj, ensure_ascii=False).encode('utf8')
            json_io = io.BytesIO(json_data)
            json_info = tarfile.TarInfo(json_file)
            json_info.size = len(json_data)
            json_info.mtime = int(time.time())
            tar.addfile(json_info, json_io)

            wav_file = key + '.' + suffix
            wav_io = io.BytesIO(data)
            wav_info = tarfile.TarInfo(wav_file)
            wav_info.size = len(data)
            wav_info.mtime = int(time.time())
            tar.addfile(wav_info, wav_io)
            write_time += (time.time() - ts)
        logging.info('read {} save {} write {}'.format(read_time, save_time,
                                                       write_time))


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')
    parser.add_argument('--seed', default='777', help='random seed')
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='whether to shuffle data')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('utt2dur_file', help='duration file')
    parser.add_argument('utt2phn_file', help='phone file')
    parser.add_argument('utt2spk_file', help='speaker file')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            assert len(arr) == 2
            wav_table[key] = arr[1]

    dur_table = {}
    with open(args.utt2dur_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            dur = [int(x) for x in arr[1:]]
            assert len(arr) >= 2
            dur_table[key] = dur

    phone_table = {}
    with open(args.utt2phn_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            assert len(arr) >= 2
            phone_table[key] = arr[1:]

    speaker_table = {}
    with open(args.utt2spk_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            assert len(arr) == 2
            speaker_table[key] = arr[1]

    data = []
    for key, wav in wav_table.items():
        if key not in dur_table or key not in phone_table:
            continue
        assert key in speaker_table
        data.append(
            (key, wav, dur_table[key], phone_table[key], speaker_table[key]))

    if args.shuffle:
        random.shuffle(data)

    num = args.num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    # Using thread pool to speedup
    pool = multiprocessing.Pool(processes=args.num_threads)
    shards_list = []
    tasks_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir,
                                '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)
        pool.apply_async(write_tar_file, (chunk, tar_file, i, num_chunks))

    pool.close()
    pool.join()

    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')


if __name__ == '__main__':
    main()
