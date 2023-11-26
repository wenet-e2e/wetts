import re
import shutil
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tritonclient.grpc as grpcclient
from pydub import AudioSegment
from scipy.io import wavfile
from stqdm import stqdm
from tritonclient.utils import np_to_triton_dtype

CACHE_DIR = Path("temp")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_TEXTS = """\
今天天气不错,我们一起去爬山
今天天气不好,我们家里躺平吧
"""


def tts(url, text, speaker, idx, verbose=False, model_name="tts", sampling_rate=24000):
    with grpcclient.InferenceServerClient(url=url, verbose=verbose) as triton_client:
        audio_text = text.strip()
        input0_data = np.array([[audio_text]], dtype=object)
        # input1_data = np.array([[speaker]], dtype=object)
        inputs = [
            grpcclient.InferInput(
                "text", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            # grpcclient.InferInput(
            #     "speaker",
            #     input1_data.shape,
            #     np_to_triton_dtype(input1_data.dtype),
            # ),
        ]

        inputs[0].set_data_from_numpy(input0_data)
        # inputs[1].set_data_from_numpy(input1_data)
        outputs = [grpcclient.InferRequestedOutput("wav")]
        start = time.time()
        response = triton_client.infer(
            model_name,
            inputs,
            outputs=outputs,
            request_id=str(idx),
        )
        end = time.time()
        result = response.as_numpy("wav")
        result = np.squeeze(result)
        duration = result.shape[0] / sampling_rate
        rtf = (end - start) / duration
        print(f"RTF: {rtf:.3f}, duration: {duration:.2f}s")
    return result


def download_widget(filepath: str, file_name: str, label="Download", place_holder=None):
    """https://docs.streamlit.io/en/stable/api.html#streamlit.download_button"""
    with open(filepath, "rb") as f:
        if not place_holder:
            place_holder = st
        btn = place_holder.download_button(
            label=label,
            data=f,
            file_name=file_name,
        )


def concat_wav(wavdir, out_wav_path):
    wavlist = list(Path(wavdir).glob("**/*.wav"))
    wavlist.sort(key=lambda x: int(re.findall(r"[0-9]+", str(x.name))[0]))
    audios = [AudioSegment.from_wav(x) for x in wavlist]
    combine_wav = audios[0]
    for x in audios[1:]:
        combine_wav += x
    combine_wav.export(out_wav_path, format="wav")


def tts_page():
    st.markdown("# TTS Vits Vocos 24k")
    speakers = "default"

    sample_rate = 24000
    server_url = st.text_input("Server URL", value="127.0.0.1:8001")
    is_concat_wav = st.checkbox("Return Concatenated Wav")
    texts = st.text_area("Your Text", value=DEFAULT_TEXTS, height=200)

    syn_button = st.button("Synthesize")
    download_button = st.empty()

    if syn_button:
        nowtime = time.strftime(r"%Y-%m-%d_%H-%M-%S-%f", time.localtime())

        if is_concat_wav:
            regex = r"([。？！!?:；：]+)"
            split_text = re.split(regex, texts)
            texts = [x.strip() for x in split_text if x.strip()]
            texts = ["".join(x) for x in zip(texts[::2], texts[1::2])]
        else:
            texts = [x.strip() for x in texts.split("\n") if x.strip()]

        tmp_dir = CACHE_DIR / nowtime
        tmp_dir.mkdir(parents=True, exist_ok=True)
        total_sen_num = len(speakers.split()) * len(texts)
        with stqdm(total=total_sen_num) as pbar:
            for speaker in speakers.split():
                try:
                    for idx, text in enumerate(texts):
                        audio = tts(server_url, text, speaker, idx, verbose=False)

                        wavpath = tmp_dir / f"{speaker}_{idx}.wav"

                        wavfile.write(
                            wavpath,
                            sample_rate,
                            audio.astype(np.int16),
                        )

                        pbar.update(1)
                        if wavpath:
                            if not is_concat_wav:
                                st.write(f"{speaker}: {text}")
                                audio_bytes = open(wavpath, "rb").read()
                                st.audio(audio_bytes, format="audio/wav", start_time=0)
                        else:
                            st.error(f"{speaker}, Syn failed at `{text}` ")
                except Exception as e:
                    st.error("Syn failed")
                    st.write(e)

        if is_concat_wav:
            download_path = CACHE_DIR / f"{nowtime}_concat.wav"
            concat_wav(tmp_dir, download_path)

            audio_bytes = open(download_path, "rb").read()
            st.audio(audio_bytes, format="audio/wav", start_time=0)
        else:  # save as zip
            with open(f"{tmp_dir}/text.txt", "w") as wf:
                for index, line in enumerate(texts, 1):
                    wf.write(f"{index}, {line}\n")
            download_path = CACHE_DIR / f"{nowtime}.zip"
            shutil.make_archive(str(tmp_dir).strip(".zip"), "zip", tmp_dir)

        download_widget(
            download_path,
            download_path.name,
            "Download",
            place_holder=download_button,
        )


if __name__ == "__main__":
    tts_page()
