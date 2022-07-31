#!/usr/bin/env bash
# Copyright 2022 Zhengxi Liu (xcmyz@outlook.com)

. path.sh

stage=                            # start from -1 if you need to download data
stop_stage=

dataset_dir=                      # path to dataset directory

fastspeech2_outputdir=fastspeech2
fastspeech2_config=conf/fastspeech2.yaml

hifigan_outputdir=hifigan
hifigan_config=conf/hifigan_v1.yaml

conda_base=$(conda info --base)
source $conda_base/bin/activate wetts


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Prepare samples
  python examples/BZNSYP/local/parse_data.py \
      --data_path $dataset_dir \
      --save_path $fastspeech2_outputdir \
      --val_samples 20 \
      --test_samples 20 \
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Compute mel, f0, energy CMVN
  total=$(cat $fastspeech2_outputdir/train/datalist.jsonl | wc -l)
  python wetts/bin/compute_cmvn.py --total $total --num_workers 32 \
      $fastspeech2_config $fastspeech2_outputdir/train/datalist.jsonl \
      $fastspeech2_outputdir/train
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # train fastspeech2
  EPOCH=
  python wetts/bin/train.py fastspeech2 --num_workers 32 \
      --config $fastspeech2_config \
      --train_data_list $fastspeech2_outputdir/train/datalist.jsonl \
      --val_data_list $fastspeech2_outputdir/val/datalist.jsonl \
      --cmvn_dir $fastspeech2_outputdir/train \
      --spk2id_file examples/BZNSYP/local/spk2id \
      --phn2id_file examples/BZNSYP/local/phn2id \
      --special_tokens_file examples/BZNSYP/local/special_token.txt \
      --log_dir log/ \
      --batch_size 64 \
      --epoch $EPOCH
fi

FASTSPEECH2_INFERENCE_OUTPUTDIR=$fastspeech2_outputdir/inference_mels # path to directory for inferenced mels
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # inference fastspeech2
  TEXT_FILE=test_samples.txt                 # path to text file, each line contains one sample for inference
  SPEAKER_FILE=test_samples_speakers.txt     # path to speaker file, each line contains one speaker name for corresponding line in text file
  FASTSPEECH2_CKPT_PATH=                     # path to fastspeech2 checkpoint file
  python wetts/bin/inference.py fastspeech2 \
      --num_workers 4 \
      --batch_size 64 \
      --config $fastspeech2_config \
      --text_file $TEXT_FILE \
      --speaker_file $SPEAKER_FILE \
      --lexicon_file $fastspeech2_outputdir/lexicon.txt \
      --cmvn_dir $fastspeech2_outputdir/train \
      --spk2id_file $fastspeech2_outputdir/spk2id \
      --phn2id_file $fastspeech2_outputdir/phn2id \
      --special_token_file $fastspeech2_outputdir/special_token.txt \
      --export_dir $FASTSPEECH2_INFERENCE_OUTPUTDIR \
      --ckpt $FASTSPEECH2_CKPT_PATH
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # train hifigan using fastspeech2 training dataset
  EPOCH=                              # Number of epoch for training hifigan
  python wetts/bin/hifigan_train.py train \
      --num_workers 32 \
      --batch_size_hifigan 32 \
      --fastspeech2_train_datalist $fastspeech2_outputdir/train/datalist.jsonl \
      --fastspeech2_val_datalist $fastspeech2_outputdir/val/datalist.jsonl \
      --hifigan_config $hifigan_config \
      --epoch $EPOCH \
      --export_dir $hifigan_outputdir/train
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # finetune hifigan using fastspeech2 training dataset
  FASTSPEECH2_CKPT_PATH=          # path to fastspeech2 checkpoint
  HIFIGAN_CKPT_PATH=              # path to hifigan generator and discriminator checkpoint
                                  # e.g. $HIFIGAN_CKPT_PATH='g_02500000 do_02500000'
                                  # pretrained hifigan checkpoint can be obtained from:
                                  # https://github.com/jik876/hifi-gan

  EPOCH=                          # number of epoch for finetune
  python wetts/bin/hifigan_train.py finetune \
      --num_workers 32 \
      --batch_size_hifigan 32 \
      --batch_size_fastspeech2 32 \
      --fastspeech2_config conf/fastspeech2.yaml \
      --fastspeech2_train_datalist $fastspeech2_outputdir/train/datalist.jsonl \
      --hifigan_config $hifigan_config \
      --phn2id_file $fastspeech2_outputdir/phn2id \
      --spk2id_file $fastspeech2_outputdir/spk2id \
      --special_tokens_file $fastspeech2_outputdir/special_token.txt \
      --cmvn_dir $fastspeech2_outputdir/train \
      --fastspeech2_ckpt $FASTSPEECH2_CKPT_PATH \
      --hifigan_ckpt $HIFIGAN_CKPT_PATH \
      --epoch $EPOCH \
      --export_dir $hifigan_outputdir/finetune
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # hifigan inference
  HIFIGAN_GENERATOR_CKPT_PATH=    # path to hifigan generator checkpoint
                                  # e.g. $HIFIGAN_GENERATOR_CKPT_PATH=g_02500000
                                  # pretrained hifigan checkpoint can be obtained from:
                                  # https://github.com/jik876/hifi-gan

  python wetts/bin/hifigan_inference.py \
      --num_workers 4 \
      --batch_size 32 \
      --config $hifigan_config \
      --datalist $FASTSPEECH2_INFERENCE_OUTPUTDIR/fastspeech2_mel_prediction.jsonl \
      --ckpt $HIFIGAN_GENERATOR_CKPT_PATH \
      --export_dir $hifigan_outputdir/wavs
fi
