import json
from pathlib import Path
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
from pypinyin import Style, lazy_pinyin
from tn.chinese.normalizer import Normalizer

MODEL_REPO = Path("/models")


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

        self.dtype = np.int64

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
            if pinyin == 'n2':
                pinyin = 'en2'

            if pinyin[-1] in "12345" and pinyin in self.pinyin_lexicon:
                phoneme_seq += self.pinyin_lexicon[pinyin]
                phoneme_seq += ['#0']
            elif pinyin in ',':
                phoneme_seq += ['#3']
            elif pinyin in '.!?':
                phoneme_seq += ['#4']
            else:
                print(f"Not valid pinyin {pinyin}", flush=True)
        phoneme_seq = ["sil"] + phoneme_seq + ["sil"]
        print(f"{text=}\n{phoneme_seq=}", flush=True)
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
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            batch_count.append(len(input0))
            for li in input0:
                total_text.append(li.decode("utf-8"))

        seqs = [self.tokenize(text) for text in total_text]
        max_seq_length = max(len(x) for x in seqs)

        input_ids = np.zeros((len(total_text), max_seq_length), dtype=self.dtype)
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
        [audios] = infer_model("decoder", [z, g], ["z", "g"], ["output"])
        audios *= 32767.0 / max(0.01, np.max(np.abs(audios))) * 0.6
        audios = np.clip(audios, -32767.0, 32767.0)
        assert audios.shape[0] == len(total_text)

        responses = []
        start = 0
        for batch in batch_count:
            sub_audios = audios[start : start + batch]
            out0 = pb_utils.Tensor("wav", sub_audios.astype(np.int16))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)
            start += batch

        assert len(requests) == len(responses)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
