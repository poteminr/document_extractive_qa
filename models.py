import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering


class BaselineModel(nn.Module):
    def __init__(self, encoder_model: str = 'cointegrated/rubert-tiny2'):
        super(BaselineModel, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(encoder_model)
        
    def forward(self, input_ids, attention_mask, start_positions, end_positions):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True
        )
        return outputs