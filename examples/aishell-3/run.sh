#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com)

stage=0 # start from -1 if you need to download data
stop_stage=5

data_url=https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz
data_dir=/mnt/mnt-data-1/binbin.zhang/data
data=data

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Download data
  local/download_data.sh $data_url $data_dir
fi

