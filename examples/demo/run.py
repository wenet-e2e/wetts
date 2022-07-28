import gradio as gr

import yacs.config
import numpy as np
import torch
from pypinyin import lazy_pinyin, Style
from scipy.io import wavfile

from wetts.models.am.fastspeech2.fastspeech2 import FastSpeech2
from wetts.bin.fastspeech2_train import load_ckpt
from wetts.models.vocoder.hifigan.hifigan import Generator as HiFiGanGenerator
from wetts.utils import file_utils
from wetts.utils.file_utils import read_key2id, read_lists, read_lexicon


model_dir = sys.argv[1]  # model dir
am_conf_path = "{}/fastspeech2.yaml".format(model_dir)
phn2id_path = "{}/phn2id".format(model_dir)
spk2id_path = "{}/spk2id".format(model_dir)
mel_cmvn_path = "{}/train/mel_cmvn.txt".format(model_dir)
pitch_cmvn_path = "{}/train/pitch_cmvn.txt".format(model_dir)
energy_cmvn_path = "{}/train/energy_cmvn.txt".format(model_dir)
am_path = "{}/fastspeech2_75.ckpt".format(model_dir)

vocoder_conf_path = "{}/hifigan_v1.yaml".format(model_dir)
vocoder_model_path = "{}/g_02500000".format(model_dir)

lexicon_path = "{}/lexicon.txt".format(model_dir)
special_token_path = "{}/special_token.txt".format(model_dir)

with open(am_conf_path) as fin:
    am_conf = yacs.config.load_cfg(fin)
phn2id = read_key2id(phn2id_path)
spk2id = read_key2id(spk2id_path)
mel_stats = np.loadtxt(mel_cmvn_path)
mel_mean = torch.from_numpy(mel_stats[0]).cuda()
mel_sigma = torch.from_numpy(mel_stats[1]).cuda()
pitch_stats = np.loadtxt(pitch_cmvn_path)
pitch_mean, pitch_sigma, pitch_min, pitch_max = pitch_stats
energy_stats = np.loadtxt(energy_cmvn_path)
energy_mean, energy_sigma, energy_min, energy_max = energy_stats

am = FastSpeech2(
    am_conf.model.d_model,
    am_conf.model.n_enc_layer,
    am_conf.model.n_enc_head,
    am_conf.model.n_enc_conv_filter,
    am_conf.model.enc_conv_kernel_size,
    am_conf.model.enc_dropout,
    len(phn2id),
    am_conf.model.padding_idx,
    am_conf.model.n_va_conv_filter,
    am_conf.model.va_conv_kernel_size,
    am_conf.model.va_dropout,
    pitch_min,
    pitch_max,
    pitch_mean,
    pitch_sigma,
    energy_min,
    energy_max,
    energy_mean,
    energy_sigma,
    am_conf.model.n_pitch_bin,
    am_conf.model.n_energy_bin,
    am_conf.model.n_dec_layer,
    am_conf.model.n_dec_head,
    am_conf.model.n_dec_conv_filter,
    am_conf.model.dec_conv_kernel_size,
    am_conf.model.dec_dropout,
    am_conf.n_mels,
    len(spk2id),
    am_conf.model.postnet_kernel_size,
    am_conf.model.postnet_hidden_dim,
    am_conf.model.n_postnet_conv_layers,
    am_conf.model.postnet_dropout,
    am_conf.model.max_pos_enc_len,
)
state_dict, *_, epoch = load_ckpt(am_path)
am.load_state_dict(state_dict)
am = am.cuda()
am.eval()

with open(vocoder_conf_path) as fin:
    vocoder_conf = yacs.config.load_cfg(fin)
vocoder = HiFiGanGenerator(
    vocoder_conf.model.resblock_kernel_sizes,
    vocoder_conf.model.resblock_dilation_sizes,
    vocoder_conf.model.upsample_rates,
    vocoder_conf.model.upsample_kernel_sizes,
    vocoder_conf.model.upsample_initial_channel,
    vocoder_conf.model.resblock_type,
)
state_dict = file_utils.load_ckpt(vocoder_model_path)
vocoder.load_state_dict(state_dict["generator"])
vocoder = vocoder.cuda()
vocoder.eval()
vocoder.remove_weight_norm()

lexicon = read_lexicon(lexicon_path)
special_tokens = set(read_lists(special_token_path))
speakers = []
for line in open(spk2id_path):
    speakers.append(line.split()[0])


def apply_lexicon(text):
    new_text = []
    for token in text:
        if token in lexicon:
            new_text.extend(lexicon[token])
        elif token in special_tokens:
            new_text.append(token)
        else:
            raise ValueError("Token {} not in lexicon or special tokens!")
    return new_text


def tts(text, speaker):
    name = "static/{}-{}.wav".format(speaker, text)
    pinyin = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    phonemes = apply_lexicon(pinyin)
    res = (pinyin, phonemes, name)

    text = [phn2id[x] for x in phonemes]
    types = [0 if x in special_tokens else 1 for x in phonemes]
    speaker = spk2id[speaker]
    with torch.no_grad():
        length = torch.tensor([len(text)], dtype=torch.int32).cuda()
        types = torch.tensor([types], dtype=torch.int32).cuda()
        speaker = torch.tensor([speaker], dtype=torch.int32).cuda()
        text = torch.tensor([text], dtype=torch.int32).cuda()

        d_control = 1.0
        (_, postnet_mel, _, mel_len, _, _, log_duration, _) = am(
            text,
            length,
            types,
            p_control=1.0,
            d_control=d_control,
            e_control=1.0,
            speaker=speaker,
        )
        postnet_mel = (postnet_mel * mel_sigma + mel_mean).permute(0, 2, 1)
        wav_pred = vocoder(postnet_mel.float()).squeeze(1)
        dur = torch.round(torch.exp(log_duration) - 1).clamp(min=0) * d_control

        wav = wav_pred[0][: int(mel_len[0]) * am_conf.hop_length].cpu().numpy()
        dur = dur[0].int().cpu().numpy()
        wavfile.write(name, vocoder_conf.sr, (wav * 32768).astype("int16"))

    return (res[0], list(zip(res[1], dur)), res[2])


demo = gr.Interface(
    fn=tts,
    inputs=[
        gr.Textbox(label="请输入文本：", placeholder="暂不支持标点符号和英文..."),
        gr.Dropdown(speakers, label="请选择说话人 ID："),
    ],
    outputs=[
        gr.Textbox(label="拼音序列："),
        gr.Textbox(label="音素时长（帧）序列："),
        gr.Audio(label="合成效果："),
    ],
)

demo.launch(server_name="0.0.0.0")
