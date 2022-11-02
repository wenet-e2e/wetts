import triton_python_backend_utils as pb_utils
import numpy as np

import json

from pypinyin import lazy_pinyin, Style
from tn.chinese.normalizer import Normalizer


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
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # TODO: read dtype from config
        self.dtype = np.int64

        self.init_dicts(self.model_config['parameters'])
        self.text_normalizer = Normalizer()
        print("Finish Init")

    def init_dicts(self, parameters):
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        # TODO: parse add_blank from config file
        self.add_blank = False
        self.blank_id = 0

        self.token_dict = {}
        with open(parameters['token_dict']) as p_f:
            for line in p_f:
                phone_id = line.strip().split()
                self.token_dict[phone_id[0]] = int(phone_id[1])

        self.pinyin_lexicon = {}
        with open(parameters['pinyin_lexicon'], 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                self.pinyin_lexicon[arr[0]] = arr[1:]

    def tokenize(self, text):
        text = self.text_normalizer.normalize(text)
        pinyin_seq = lazy_pinyin(
            text,
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors=lambda punct_and_en_words: list(punct_and_en_words),
        )
        phoneme_seq = []
        for pinyin in pinyin_seq:
            # 12345 represents five tones
            if pinyin[-1] in '12345':
                assert pinyin in self.pinyin_lexicon
                phoneme_seq += self.pinyin_lexicon[pinyin]
            else:
                # Pinyins would end up with a number in 1-5,
                # which represents tones of the pinyin.
                # For symbols which are not pinyin,
                # e.g. English letters, Chinese puncts, we directly use them as inputs.
                phoneme_seq.append(pinyin)
        seq = [self.token_dict[symbol] for symbol in phoneme_seq]
        if self.add_blank:
            import commons
            seq = commons.intersperse(seq, self.blank_id)
        return seq

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        batch_count = []
        total_text = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(
                request, "text").as_numpy()
            batch_count.append(len(input0))
            for li in input0:
                total_text.append(li.decode("utf-8"))

        seqs = [self.tokenize(text) for text in total_text]
        max_seq_length = max(len(x) for x in seqs)

        input_ids = np.zeros(
            (len(total_text), max_seq_length), dtype=self.dtype)
        input_lengths = np.zeros(len(total_text), dtype=np.int64)
        # hard-coding scales here, may accept them from client
        # noise scale, length scale, noise_scale_w
        input_scales = np.tile(
            np.array([0.667, 1.0, 0.8], dtype=np.float32), (len(total_text), 1))

        for i, seq in enumerate(seqs):
            input_ids[i][:len(seq)] = seq
            input_lengths[i] = len(seq)
        input_lengths = np.expand_dims(input_lengths, axis=1)

        in_0 = pb_utils.Tensor("input", input_ids)
        in_1 = pb_utils.Tensor("input_lengths", input_lengths)
        in_2 = pb_utils.Tensor("scales", input_scales)

        inference_request = pb_utils.InferenceRequest(
            model_name='generator',
            requested_output_names=['output'],
            inputs=[in_0, in_1, in_2])

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            audios = pb_utils.get_output_tensor_by_name(inference_response,
                                                        'output')
            audios = audios.as_numpy()
            audios *= 32767.0 / max(0.01, np.max(np.abs(audios))) * 0.6
            audios = np.clip(audios, -32767.0, 32767.0)
            assert audios.shape[0] == len(total_text)

        responses = []
        start = 0
        for batch in batch_count:
            sub_audios = audios[start:start + batch]
            out0 = pb_utils.Tensor("wav", sub_audios.astype(np.int16))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0])
            responses.append(inference_response)
            start += batch

        assert len(requests) == len(responses)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
