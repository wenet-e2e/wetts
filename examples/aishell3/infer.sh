#!/usr/bin/env bash
# Copyright (c) 2022 Tsinghua University(Jie Chen)

. path.sh

NUM_WORKERS=2                   # number of workers for dataloader
BATCH_SIZE=64                   # batch size for dataloader
CONFIG=conf/default.yaml        # model config file
TEXT_FILE=                      # file containing text
SPEAKER_FILE=                   # file containing speakers
LEXICON_FILE=                   # file containing lexicons
CMVN_DIR=                       # cmvn directory
SPK2ID_FILE=                    # spk2id file
PHN2ID_FILE=                    # phn2id file
SPECIAL_TOKEN_FILE=             # file containing special tokens
EXPORT_DIR=                     # path to directory saving generated mel
CKPT=                           # path to fastspeech2 checkpoint

conda_base=$(conda info --base)
source $conda_base/bin/activate wetts
python wetts/bin/inference.py fastspeech2 --num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --config $CONFIG \
    --text_file $TEXT_FILE \
    --speaker_file $SPEAKER_FILE \
    --lexicon_file $LEXICON_FILE \
    --cmvn_dir $CMVN_DIR \
    --spk2id_file $SPK2ID_FILE \
    --phn2id_file $PHN2ID_FILE \
    --special_token_file $SPECIAL_TOKEN_FILE \
    --export_dir $EXPORT_DIR \
    --ckpt $CKPT_PATH
conda deactivate
