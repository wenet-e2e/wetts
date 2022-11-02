# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import multiprocessing
from multiprocessing import Pool
import os
import argparse
from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import numpy as np
from scipy.io import wavfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is '
                        '10.19.222.12:8001.')
    parser.add_argument('--model_name',
                        required=False,
                        default='tts',
                        help='the model to send request to')
    parser.add_argument('--text',
                        type=str,
                        required=False,
                        default=None,
                        help='each line is a text scp file')
    parser.add_argument('--outdir', required=True,
                        help='ouput wav file directory')
    parser.add_argument('--sampling_rate', type=int, required=False,
                        default="22050", help='wav file sampling rate')

    FLAGS = parser.parse_args()

    total_lines = []
    # load data
    with open(FLAGS.text, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            total_lines.append(line)

    num_workers = min(multiprocessing.cpu_count() // 2, len(total_lines))

    def single_job(client_textlines):
        predictions = []
        with grpcclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:
            idx, textlines = client_textlines
            for cur_id, li in enumerate(textlines):
                audio_name, audio_text = li.strip().split("|", 1)
                audio_name = audio_name.strip()
                input0_data = np.array([[audio_text]], dtype=object)
                inputs = [
                    grpcclient.InferInput(
                        "text", input0_data.shape, np_to_triton_dtype(input0_data.dtype))
                ]

                inputs[0].set_data_from_numpy(input0_data)
                outputs = [
                    grpcclient.InferRequestedOutput("wav")
                ]
                response = triton_client.infer(FLAGS.model_name,
                                               inputs,
                                               outputs=outputs,
                                               request_id=str(idx + cur_id))
                result = response.as_numpy("wav")
                result = np.squeeze(result)
                # TODO: remove audio padding here, assert dtype np.int16 here
                predictions.append((audio_name, result))
        return predictions

    # start to do inference
    # Group requests in batches
    tasks = []
    splits = np.array_split(total_lines, num_workers)

    for idx, per_split in enumerate(splits):
        cur_files = per_split.tolist()
        tasks.append((idx, cur_files))

    with Pool(processes=num_workers) as pool:
        predictions = pool.map(single_job, tasks)

    predictions = [item for sub_pred in predictions for item in sub_pred]

    os.makedirs(FLAGS.outdir, exist_ok=True)

    for audio_name, result in predictions:
        assert len(result.shape) == 1, result.shape
        wavfile.write(FLAGS.outdir + "/" + audio_name + ".wav",
                      FLAGS.sampling_rate, result.astype(np.int16))
