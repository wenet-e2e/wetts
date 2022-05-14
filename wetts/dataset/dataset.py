# Copyright (c) 2022 Horizon Robtics. (authors: Binbin Zhang)
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

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import wetts.dataset.processor as processor
from wetts.utils.file_utils import read_key2id, read_lists


class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def CmvnDataset(data_list_file, conf):
    lists = read_lists(data_list_file)
    # Global shuffle
    dataset = DataList(lists, shuffle=False)
    dataset = Processor(dataset, processor.url_opener)
    dataset = Processor(dataset, processor.tar_file_and_group)
    dataset = Processor(dataset, processor.compute_feats, conf)
    return dataset


def Dataset(data_list_file, spk2id_file, phn2id_file, cmvn_dir, conf):
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', False)
    # Global shuffle
    dataset = DataList(lists, shuffle=shuffle)
    dataset = Processor(dataset, processor.url_opener)
    dataset = Processor(dataset, processor.tar_file_and_group)
    # Local shuffle
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)
    # Apply speaker/phone mapping
    spk2id = read_key2id(spk2id_file)
    phn2id = read_key2id(phn2id_file)
    dataset = Processor(dataset, processor.apply_spk2id, spk2id)
    dataset = Processor(dataset, processor.apply_phn2id, phn2id)

    dataset = Processor(dataset, processor.compute_feats, conf)
    # CMVN
    mel_stats = np.loadtxt(os.path.join(cmvn_dir, 'mel_cmvn.txt'))
    f0_stats = np.loadtxt(os.path.join(cmvn_dir, 'f0_cmvn.txt'))
    energy_stats = np.loadtxt(os.path.join(cmvn_dir, 'energy_cmvn.txt'))
    dataset = Processor(dataset, processor.apply_cmvn, mel_stats, f0_stats,
                        energy_stats)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset
