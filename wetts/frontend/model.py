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

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class FrontendModel(nn.Module):

    def __init__(self,
                 num_polyphones: int,
                 num_prosody: int,
                 bert_name_or_path: str = 'bert-chinese-base'):
        super(FrontendModel, self).__init__()
        print(bert_name_or_path)
        self.bert = AutoModel.from_pretrained(bert_name_or_path)
        for param in self.bert.parameters():
            param.requires_grad_(False)

        if 'bert-chinese-base' in bert_name_or_path:
            d_model = 768
            nhead = 8
            dim_feedforward = 2048
        elif 'TinyBERT_4L_zh_backup' in bert_name_or_path:
            d_model = 312
            nhead = 12
            dim_feedforward = 1200
        else:
            assert False, f'unsupport model {bert_name_or_path}'
        self.transform = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        self.phone_classifier = nn.Linear(d_model, num_polyphones)
        self.prosody_classifier = nn.Linear(d_model, num_prosody)

    def _forward(self, x):
        mask = x["attention_mask"] == 0
        bert_output = self.bert(**x)
        x = self.transform(bert_output.last_hidden_state,
                           src_key_padding_mask=mask)
        phone_pred = self.phone_classifier(x)
        prosody_pred = self.prosody_classifier(x)
        return phone_pred, prosody_pred

    def forward(self, x):
        return self._forward(x)

    def export_forward(self, x):
        assert x.size(0) == 1
        x = {
            "input_ids": x,
            "token_type_ids": torch.zeros(1, x.size(1), dtype=torch.int64),
            "attention_mask": torch.ones(1, x.size(1), dtype=torch.int64),
        }
        phone_logits, prosody_logits = self._forward(x)
        phone_pred = F.softmax(phone_logits, dim=-1)
        prosody_pred = F.softmax(prosody_logits, dim=-1)
        return phone_pred, prosody_pred
