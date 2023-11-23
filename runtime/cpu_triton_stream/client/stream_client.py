#!/usr/bin/python3
import argparse
import queue
import random
import time
from functools import partial
from pathlib import Path
from typing import Iterator, List, TypeVar

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

T = TypeVar("T", bound=grpcclient.InferResult)


class UserData:
    def __init__(self):
        self._completed_requests: queue.Queue[T] = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class StreamGrpcTritonClient:
    """simply triton client"""

    def __init__(self, url: str = "localhost:8001"):
        self.client = grpcclient.InferenceServerClient(url, verbose=False)

    def send_request(
        self,
        model_name: str,
        inputs: List[np.ndarray],
        inputs_names: List[str],
        outputs_names: List[str],
        model_version: str = "",
        request_id="",
        sequence_id=0,
        sequence_start=False,
        sequence_end=False,
        priority=0,
        timeout: int = 1,
    ) -> Iterator[grpcclient.InferResult]:
        triton_inputs = []
        for input, name in zip(inputs, inputs_names):
            triton_input = grpcclient.InferInput(
                name, input.shape, np_to_triton_dtype(input.dtype)
            )
            triton_input.set_data_from_numpy(input)
            triton_inputs.append(triton_input)

        triton_outputs = [grpcclient.InferRequestedOutput(x) for x in outputs_names]

        user_data = UserData()

        self.client.start_stream(callback=partial(callback, user_data))
        self.client.async_stream_infer(
            model_name=model_name,
            inputs=triton_inputs,
            request_id=request_id,
            outputs=triton_outputs,
            model_version=model_version,
            sequence_id=sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            priority=priority,
            timeout=timeout,
            enable_empty_final_response=True,  # To get the end signal
        )

        # Send None to indicate that the send end request is complete
        self.client._stream._enqueue_request(None)

        while True:
            try:
                data_item = user_data._completed_requests.get()
                if type(data_item) == InferenceServerException:
                    print(f"error: {data_item}")
                    raise data_item
                else:
                    triton_final_response: bool = data_item.get_response(as_json=True)[
                        "parameters"
                    ]["triton_final_response"]["bool_param"]
                    if triton_final_response:
                        self.client.stop_stream()
                        break

                    yield data_item
            except queue.Empty:
                break


def strs2array(strs: List[str]):
    return np.array(strs, dtype=np.object_)


def rand_int_string(length: int = 15) -> str:
    return "".join([str(random.randint(0, 9)) for i in range(length)])


class StreamTTSClient:
    def __init__(self, url, model_name, sr) -> None:
        self.client = StreamGrpcTritonClient(url)
        self.sr = sr
        self.model_name = model_name

    def text2audio(self, text, output_wavpath):
        outputs = []
        chunk_id = 0
        st = time.time()
        dur = 0
        latency = 0
        first_latency = 0

        for response in self.stream_tts(text):
            et = time.time()
            audio_chunk = response.as_numpy("wav")
            chunk_duration = audio_chunk.shape[0] / self.sr
            chunk_latency = et - st
            print(f"{chunk_id=}, {chunk_latency=:.2f}, {chunk_duration=:.2f}s")

            if chunk_id == 0:
                first_latency = chunk_latency
            st = et
            chunk_id += 1
            outputs.append(audio_chunk)
            latency += chunk_latency
            infer_time = latency - first_latency
            if infer_time > dur:
                print(f"Error! infer_time > dur, {infer_time=:.2f}, {dur=:.2f}")
            dur += chunk_duration

        rtf = latency / dur
        print(f"{dur=:.2f}, {rtf=:.2f}, {first_latency=:.3f}\n")

        wav = np.concatenate([x.squeeze() for x in outputs])

        sf.write(output_wavpath, wav, self.sr)

    def stream_tts(
        self,
        text: str,
        speaker: str = "baker",
        model_version="",
    ):
        text_array = strs2array([text])
        # speaker_array = strs2array([speaker])

        request_id = rand_int_string()
        return self.client.send_request(
            self.model_name,
            [text_array],
            ["text"],
            ["wav"],
            model_version=model_version,
            request_id=request_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is " "10.19.222.12:8001.",
    )
    parser.add_argument(
        "--model_name",
        required=False,
        default="stream_tts",
        help="the model to send request to",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        default=None,
        help="each line is a text scp file",
    )
    parser.add_argument("--outdir", required=True, help="ouput wav file directory")
    parser.add_argument(
        "--sampling_rate",
        type=int,
        required=False,
        default="24000",
        help="wav file sampling rate",
    )

    FLAGS = parser.parse_args()

    outdir = Path(FLAGS.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tts_client = StreamTTSClient(FLAGS.url, FLAGS.model_name, FLAGS.sampling_rate)

    total_lines = []

    with open(FLAGS.text, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            total_lines.append(line)

    for line in total_lines:
        print(line)
        audio_name, audio_text = line.strip().split("|", 1)
        tts_client.text2audio(audio_text, outdir / (audio_name + ".wav"))
