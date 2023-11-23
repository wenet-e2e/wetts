import json
import math
from pathlib import Path
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
from pypinyin import Style, lazy_pinyin
from tn.chinese.normalizer import Normalizer

MODEL_REPO = Path("/models")
MIN_CHUNK = 65
VOC_BLOCK_SIZE = 70
VOC_PAD_SIZE = 10
UPSAMPLE_SIZE = 256
SAMPLE_RATE = 24000


def compose(
    model_name: str,
    inputs: List[np.ndarray],
    inputs_names: List[str],
    outputs_names: List[str],
):
    input_tensors = []
    for input, input_name in zip(inputs, inputs_names):
        input_tensors.append(pb_utils.Tensor(input_name, input))

    infer_request = pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=outputs_names,
        inputs=input_tensors,
    )
    return infer_request


def get_output_tensor(infer_response, outputs_names: List[str]) -> List[np.ndarray]:
    if infer_response.has_error():
        raise pb_utils.TritonModelException(infer_response.error().message())
    outputs = []
    for output_name in outputs_names:
        output_tensor = pb_utils.get_output_tensor_by_name(infer_response, output_name)
        output = output_tensor.as_numpy()
        outputs.append(output)
    return outputs


def infer_model(
    model_name: str,
    inputs: List[np.ndarray],
    inputs_names: List[str],
    outputs_names: List[str],
) -> List[np.ndarray]:
    infer_request = compose(model_name, inputs, inputs_names, outputs_names)
    infer_response = infer_request.exec()
    return get_output_tensor(infer_response, outputs_names)


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

    mel_len = mel.shape[-1]

    chunks = []
    n = math.ceil(mel_len / block_size)
    for i in range(n):
        start = max(0, i * block_size - pad_size)
        end = min((i + 1) * block_size + pad_size, mel_len)
        chunks.append(mel[:, :, start:end])
    pad_end = 0
    # Padding the last chunk if its size is less than MIN_CHUNK
    if chunks[-1].shape[-1] < MIN_CHUNK:
        pad_end = MIN_CHUNK - chunks[-1].shape[-1]
        chunks[-1] = np.pad(chunks[-1], ((0, 0), (0, 0), (0, pad_end)), mode="reflect")

    return chunks, pad_end


def depadding(audio, chunk_num, chunk_id, block, pad, upsample, pad_end):
    """
    Streaming inference removes the result of pad inference

    Args:
        audio: (np.ndarray): shape (B, T)
    """

    assert len(audio.shape) == 2
    front_pad = min(chunk_id * block, pad)
    # first chunk
    if chunk_id == 0:
        audio = audio[:, : block * upsample]
    # last chunk
    elif chunk_id == chunk_num - 1:
        audio = audio[
            :, front_pad * upsample : -pad_end * upsample
        ]  # Remove the added padding
    # middle chunk
    else:
        audio = audio[:, front_pad * upsample : (front_pad + block) * upsample]

    return audio


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args["model_config"])

        params = model_config["parameters"]

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        self.init_dicts(params)
        self.text_normalizer = Normalizer()
        print("Finish Init")

    def init_dicts(self, parameters):
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        self.blank_id = 0

        self.token_dict = {}

        # with open(MODEL_REPO / f"base.json") as j_f:
        #     self.base_config = json.load(j_f)

        with open(MODEL_REPO / "phones.txt") as p_f:
            for line in p_f:
                phone_id = line.strip().split()
                self.token_dict[phone_id[0]] = int(phone_id[1])

        self.pinyin_lexicon = {}
        with open(MODEL_REPO / "lexicon.txt", "r", encoding="utf8") as fin:
            for line in fin:
                arr = line.strip().split()
                self.pinyin_lexicon[arr[0]] = arr[1:]

    def tokenize(self, text):
        text = self.text_normalizer.normalize(text.strip())
        pinyin_seq = lazy_pinyin(
            text,
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors=lambda punct_and_en_words: list(punct_and_en_words),
        )
        # add '#0' element to the end of each pinyin
        phoneme_seq = []
        for i, pinyin in enumerate(pinyin_seq):
            # 12345 represents five tones
            if pinyin[-1] in "12345":
                assert pinyin in self.pinyin_lexicon
                phoneme_seq += self.pinyin_lexicon[pinyin]
            else:
                # Pinyins would end up with a number in 1-5,
                # which represents tones of the pinyin.
                # For symbols which are not pinyin,
                # e.g. English letters, Chinese puncts, we directly use them as inputs.
                phoneme_seq.append(pinyin)
            # add prosody boundary like baker example
            if i != len(pinyin_seq) - 1:
                phoneme_seq.append("#0")
        phoneme_seq = ["sil"] + phoneme_seq + ["#4", "sil"]
        seq = []
        for symbol in phoneme_seq:
            if symbol in self.token_dict:
                seq.append(self.token_dict[symbol])
            else:
                print(f"unknown {symbol=}", flush=True)
        return seq

    def execute(self, requests):
        batch_count = []
        total_text = []

        if len(requests) != 1:
            raise pb_utils.TritonModelException(
                "unsupported batch size " + str(len(requests))
            )

        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            batch_count.append(len(input0))
            for li in input0:
                total_text.append(li.decode("utf-8"))

        response_sender = request.get_response_sender()

        seqs = [self.tokenize(text) for text in total_text]
        max_seq_length = max(len(x) for x in seqs)

        input_ids = np.zeros((len(total_text), max_seq_length), dtype=np.int64)
        input_lengths = np.zeros(len(total_text), dtype=np.int64)
        # hard-coding scales here, may accept them from client
        # noise scale, length scale, noise_scale_w
        scales = np.tile(
            np.array([0.667, 1.0, 0.8], dtype=np.float32), (len(total_text), 1)
        )

        for i, seq in enumerate(seqs):
            input_ids[i][: len(seq)] = seq
            input_lengths[i] = len(seq)
        input_lengths = np.expand_dims(input_lengths, axis=1)
        sid = np.array([0], dtype=np.int64)
        sid = np.expand_dims(sid, axis=1)

        z, g = infer_model(
            "encoder",
            [input_ids, input_lengths, scales, sid],
            ["input", "input_lengths", "scales", "sid"],
            ["z", "g"],
        )

        z_chunks, pad_end = get_chunks(z, VOC_BLOCK_SIZE, VOC_PAD_SIZE)

        for i, z_chunk in enumerate(z_chunks):
            [audio_chunk] = infer_model("decoder", [z_chunk, g], ["z", "g"], ["output"])
            audio_chunk = depadding(
                audio_chunk.reshape(1, -1),
                len(z_chunks),
                i,
                VOC_BLOCK_SIZE,
                VOC_PAD_SIZE,
                UPSAMPLE_SIZE,
                pad_end,
            )
            audio_chunk = audio_chunk.reshape(-1)
            audio_chunk = np.clip((audio_chunk * 32767), -32767.0, 32767.0).astype(
                np.int16
            )

            # print(f"{i=} {z_chunk.shape=} {audio_chunk.shape=}", flush=True)
            audio_tensor = pb_utils.Tensor("wav", audio_chunk)
            response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
            response_sender.send(response)

        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def finalize(self):
        print("Cleaning up...")
