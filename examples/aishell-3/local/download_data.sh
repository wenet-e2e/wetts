#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com)

if [ $# -ne 2 ]; then
  echo "Usage: $0 <url> <download_dir>"
  exit 0;
fi

url=$1
dir=$2

[ ! -d $dir ] && mkdir -p $dir

# Download data
if [ ! -f $dir/data_aishell3.tgz ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  echo "$0: downloading data from $url. This may take some time, please wait"

  cd $dir
  if ! wget --no-check-certificate $url; then
    echo "$0: error executing wget $url"
    exit 1;
  fi
fi


cd $dir
if ! tar -xvzf data_aishell3.tgz; then
  echo "$0: error un-tarring archive $dir/data_aishell3.tgz"
  exit 1;
fi
