# Copyright (c) 2022, Binbin Zhang (binbzha@qq.com)
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

import torch.nn as nn

from transformers import AutoModel


class ProsodyModel(nn.Module):
    def __init__(self, num_prosody: int = 4):
        super(ProsodyModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters():
            param.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=768,
                                       nhead=8,
                                       dim_feedforward=2048,
                                       batch_first=True),
            nn.Linear(768, num_prosody),
        )

    def forward(self, x):
        bert_output = self.bert(**x)
        logits = self.classifier(bert_output.last_hidden_state)
        return logits
