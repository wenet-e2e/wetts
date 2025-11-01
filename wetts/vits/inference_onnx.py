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
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from scipy.io import wavfile
from utils import task


def to_numpy(tensor):
    return (tensor.detach().cpu().numpy()
            if tensor.requires_grad else tensor.detach().numpy())


def add_prefix(filepath, prefix):
    filepath = Path(filepath)
    return str(filepath.parent / (prefix + filepath.name))


# Copy from: runtime/cpu_triton_stream/model_repo/stream_tts/1/model.py
def get_chunks(mel, block_size, pad_size):
    """Divide mel into multiple chunks with overlap(overlap_size == pad_size)
    Args:
        mel (np.ndarray): shape (B, freq, Frame)
        block_size (int)
        pad_size (int)
    Returns:
        list: chunks list
    """
    if block_size == -1:
        return [mel]
    mel_len = mel.shape[1]
    chunks = []
    n = math.ceil(mel_len / block_size)
    for i in range(n):
        start = max(0, i * block_size - pad_size)
        end = min((i + 1) * block_size + pad_size, mel_len)
        print(start, end)
        chunks.append(mel[:, start:end, :])
    return chunks


# Copy from: runtime/cpu_triton_stream/model_repo/stream_tts/1/model.py
def depadding(audio, chunk_num, chunk_id, block, pad, upsample):
    """
    Streaming inference removes the result of pad inference

    Args:
        audio: (np.ndarray): shape (B, T)
    """

    assert len(audio.shape) == 2
    front_pad = min(chunk_id * block, pad)
    if chunk_id == 0:  # First chunk
        audio = audio[:, :block * upsample]
    elif chunk_id == chunk_num - 1:  # Last chunk
        audio = audio[:, front_pad * upsample:]  # Remove the added padding
    else:  # Middle chunk
        audio = audio[:, front_pad * upsample:(front_pad + block) * upsample]
    return audio


def get_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--onnx_model", required=True, help="onnx model")
    parser.add_argument("--cfg", required=True, help="config file")
    parser.add_argument("--outdir", required=True, help="ouput directory")
    parser.add_argument("--phone_table",
                        required=True,
                        help="input phone dict")
    parser.add_argument("--speaker_table", default=True, help="speaker table")
    parser.add_argument("--streaming",
                        action="store_true",
                        help="export streaming model")
    parser.add_argument("--test_file", required=True, help="test file")
    parser.add_argument(
        "--providers",
        required=False,
        default="CPUExecutionProvider",
        choices=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="onnx runtime providers",
    )
    parser.add_argument("--chunk_size",
                        type=int,
                        default=-1,
                        help="for streaming")
    parser.add_argument("--pad_size",
                        type=int,
                        default=0,
                        help="for streaming")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    phone_dict = {}
    for line in open(args.phone_table):
        phone_id = line.strip().split()
        phone_dict[phone_id[0]] = int(phone_id[1])

    speaker_dict = {}
    for line in open(args.speaker_table):
        arr = line.strip().split()
        assert len(arr) == 2
        speaker_dict[arr[0]] = int(arr[1])
    hps = task.get_hparams_from_file(args.cfg)
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    # make triton dynamic shape happy
    scales = scales.unsqueeze(0)

    if args.streaming:
        encoder_ort_sess = ort.InferenceSession(
            add_prefix(args.onnx_model, 'encoder_'),
            providers=[args.providers],
        )
        decoder_ort_sess = ort.InferenceSession(
            add_prefix(args.onnx_model, 'decoder_'),
            providers=[args.providers],
        )

        # Copy from: runtime/cpu_triton_stream/model_repo/stream_tts/1/model.py
        def tts(ort_inputs):
            sid = ort_inputs['sid']
            z = encoder_ort_sess.run(None, ort_inputs)[0]
            z_chunks = get_chunks(z, args.chunk_size, args.pad_size)
            num_chunks = len(z_chunks)
            audios = []
            for i, chunk in enumerate(z_chunks):
                decoder_inputs = {"z": chunk, "sid": sid}
                audio_chunk = decoder_ort_sess.run(None, decoder_inputs)[0]
                audio_clip = depadding(audio_chunk.reshape(1, -1), num_chunks,
                                       i, args.chunk_size, args.pad_size, 256)
                audios.append(audio_clip)
            return np.squeeze(np.concatenate(audios, axis=1))

    else:
        ort_sess = ort.InferenceSession(args.onnx_model,
                                        providers=[args.providers])

        def tts(ort_inputs):
            return np.squeeze(ort_sess.run(None, ort_inputs))

    for line in open(args.test_file):
        audio_path, speaker, text = line.strip().split("|")
        sid = speaker_dict[speaker]
        seq = [phone_dict[symbol] for symbol in text.split()]

        x = torch.LongTensor([seq])
        x_len = torch.IntTensor([x.size(1)]).long()
        sid = torch.LongTensor([sid]).long()
        ort_inputs = {
            "input": to_numpy(x),
            "input_lengths": to_numpy(x_len),
            "scales": to_numpy(scales),
            "sid": to_numpy(sid),
        }
        audio = tts(ort_inputs)
        audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
        audio = np.clip(audio, -32767.0, 32767.0)
        wavfile.write(
            args.outdir + "/" + audio_path.split("/")[-1],
            hps.data.sampling_rate,
            audio.astype(np.int16),
        )


if __name__ == "__main__":
    main()
