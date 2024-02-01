# Copyright (c) 2024, Shengqiang Li (shengqiang.li96@gmail.com)
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


import gradio as gr
from wetts.cli.model import load_model


def main():
    title = "End-to-End Speech Synthesis in WeTTS | 基于 WeTTS 的端到端语音合成"
    description = "WeTTS Demo"
    inputs=[gr.Textbox(label="text")]
    phones = gr.Textbox(label="phones")
    audio = gr.Audio(label="audio")
    outputs = [phones, audio]
    gr.Interface(
        synthesis,
        title=title,
        description=description,
        inputs=inputs,
        outputs=outputs
    ).launch(server_name='0.0.0.0', share=True)


def synthesis(text):
    model = load_model()
    phones, audio = model.synthesis(text)
    sampling_rate = 16000
    return ' '.join(phones), (sampling_rate, audio)


if __name__ == '__main__':
    main()
