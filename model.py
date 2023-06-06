# coding=utf-8
from typing import Optional
import torch
import torch.nn as nn
from transformers import ErniePreTrainedModel, ErnieModel, ErnieConfig
from utils import UTCLoss

class UTC(ErniePreTrainedModel):
    """
    adapted from paddlenlp.designed for unified tag classification
    """
    def __init__(self, config: ErnieConfig):
        super(UTC, self).__init__(config)
        self.ernie = ErnieModel(config)
        self.predict_size = 64
        self.linear_q = nn.Linear(config.hidden_size, self.predict_size)
        self.linear_k = nn.Linear(config.hidden_size, self.predict_size)
    def forward(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        omask_positions,
        cls_positions,
        labels: Optional[torch.Tensor] = None
    ):
        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]

        batch_size, seq_len, hidden_size = sequence_output.size()
        flat_sequence_output = torch.reshape(sequence_output, [-1, hidden_size])
        flat_length = torch.arange(batch_size, device=sequence_output.device) * seq_len
        cls_positions += flat_length
        
        cls_output = torch.index_select(flat_sequence_output, 0, cls_positions)
        
        q = self.linear_q(cls_output)
        
        option_output = torch.index_select(
            flat_sequence_output, 
            0,
            torch.reshape(omask_positions + flat_length.unsqueeze(1), [-1])
        )
        option_output = torch.reshape(option_output, [batch_size, -1, hidden_size])
        k = self.linear_k(option_output)
        

        option_logits = torch.matmul(q.unsqueeze(1), k.permute((0, 2, 1))).squeeze(1)
        option_logits = option_logits / self.predict_size**0.5
        for index, logit in enumerate(option_logits):
            option_logits[index] -= (1 - (omask_positions[index] > 0).to(torch.float32)) * 1e12

        res_outputs = {'option_logits': option_logits}
        if labels is not None:
            loss_fn = UTCLoss()
            loss = loss_fn(option_logits, labels)
            res_outputs['loss'] = loss

        return res_outputs
