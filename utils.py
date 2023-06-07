# coding=utf-8
"""
adapted from paddlenlp
"""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

import numpy as np
from log import logger
import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

class MLMTokenizerWrap(object):

    omask_token = "[O-MASK]"

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, inputs: List[Dict[str, Any]]):
        encoded_inputs = defaultdict(list)
        last_position = 1  # Id 0 denotes special token '[CLS]'.
        last_token_type = 0
        option_length = None

        orig_input_ids = []
        # 生成最初的input_ids
        for index, part in enumerate(inputs):
            # print(part)
            orig_input_ids.append(
                self.tokenizer.encode(
                    part['text'], 
                    add_special_tokens=False,
                    return_token_type_ids=False
                )
            )
        
        # 计算inputs中每个part应该保留长度
        part_do_truncate = [part['do_truncate'] for part in inputs]
        max_lengths = self._create_max_lengths_from_do_truncate(orig_input_ids, part_do_truncate)

        for index, part in enumerate(inputs):
            # 生成input_ids
            if self.tokenizer.truncation_side == "left":
                input_ids = orig_input_ids[index][-max_lengths[index] :]
            else:
                input_ids = orig_input_ids[index][: max_lengths[index]]

            encoded_inputs["input_ids"].append(input_ids)
            part_length = len(input_ids)

            # 生成position_ids
            position_ids, last_position = self._create_position_ids_from_part(input_ids, part, last_position)
            encoded_inputs["position_ids"].append(position_ids)

            # 生成token_type_ids.
            if "token_types" in part:
                last_token_type = part["token_types"]
            encoded_inputs["token_type_ids"].append([last_token_type] * part_length)

            # 生成其他特征
            for name in part:
                if name not in ["text", "positions", "token_types", "do_truncate"]:
                    encoded_inputs[name].append([part[name]] * part_length)

            # Record the length of options if exists.
            if self.omask_token in part["text"]:
                option_length = len(input_ids)

        encoded_inputs = self.join(encoded_inputs)
        encoded_inputs = self.add_special_tokens(encoded_inputs)
        attention_mask = self._create_attention_mask(encoded_inputs["input_ids"], option_length)
        if attention_mask is not None:
            encoded_inputs["attention_mask"] = attention_mask
        masked_positions = self._create_masked_positions(encoded_inputs["input_ids"])
        if masked_positions is not None:
            encoded_inputs["masked_positions"] = masked_positions
        return encoded_inputs
    
    def add_special_tokens(self, input_dict: Dict[str, Any]):
        for key in input_dict:
            new_inputs = self.tokenizer.build_inputs_with_special_tokens(input_dict[key])
            if key != "input_ids":
                special_mask = np.array(self.tokenizer.get_special_tokens_mask(input_dict[key]))
                new_inputs = np.array(new_inputs)
                new_inputs[special_mask == 1] = 0
                new_inputs = new_inputs.tolist()
            input_dict[key] = new_inputs
        return input_dict
 
    @staticmethod
    def join(input_dict: Dict[str, Any]):
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict

    def _create_attention_mask(self, input_ids: List[int], option_length: Optional[int]):
        if option_length is None:
            return None
        input_ids = np.array(input_ids)
        omask_id = self.tokenizer.convert_tokens_to_ids(self.omask_token)
        attention_mask = np.ones([len(input_ids), len(input_ids)])
        omask_index = np.where(input_ids == omask_id)[0].tolist()
        cls_indices = np.where(input_ids == self.tokenizer.cls_token_id)[0]
        sep_indices = np.where(input_ids == self.tokenizer.sep_token_id)[0]

        cls_index = len(input_ids)
        for idx in cls_indices:
            if idx > omask_index[-1]:
                cls_index = idx
                break
        sep_index = len(input_ids)
        for idx in sep_indices:
            if idx > omask_index[-1]:
                sep_index = idx
                break
        opt_begin = omask_index[0]
        opt_end = min(cls_index, sep_index)

        attention_mask[opt_begin:opt_end, opt_begin:opt_end] = 0
        omask_index.append(opt_end)
        for opt_begin, opt_end in zip(omask_index[:-1], omask_index[1:]):
            attention_mask[opt_begin:opt_end, opt_begin:opt_end] = 1
        
        return attention_mask

    def _create_masked_positions(self, input_ids: List[int]):
        mask_id = self.tokenizer.mask_token_id

        masked_positions = np.where(input_ids == mask_id)[0]
        if masked_positions.shape[0] == 0:
            return None
        return masked_positions.tolist()
    
    def _create_max_lengths_from_do_truncate(self, orig_input_ids: List[List[int]], part_do_truncate: List[bool]):
        """
        计算prompt每个part的最大长度，优先truncate长度最大的part，不被truncate的part对应None
        """
        text_length = sum([len(x) for x in orig_input_ids])
        num_special_token = self.tokenizer.num_special_tokens_to_add()
        max_length = self.max_length - num_special_token

        max_lengths = [len(part) for part in orig_input_ids]
        if text_length <= max_length:
            return max_lengths
        
        # 处理do_truncate为False的part，主要是扣除它对应的长度
        for index, part in enumerate(orig_input_ids):
            if not part_do_truncate[index]:
                max_length -= len(part)
        
        if sum(part_do_truncate) == 0:
            logger.warning(
                'f"Can not truncate the sequence with length {text_length}. Set more `truncate` attributes as True."'
            )
            return max_lengths

        # 长度小于max_length//sum(part_do_truncate)的part不做truncate
        has_short = True
        while has_short:
            has_short = False
            avg_max_length = max_length // sum(part_do_truncate)
            for index, part in enumerate(orig_input_ids):
                if part_do_truncate[index] and len(part) <= avg_max_length:
                    part_do_truncate[index] = False
                    max_length -= len(part)
                    has_short = True
            
        if max_length < 0:
            raise AssertionError('Actual length has extended the maximum length.')
        
        avg_max_length = max_length // sum(part_do_truncate)
        for index in range(len(orig_input_ids)):
            if part_do_truncate[index]:
                max_lengths[index] = avg_max_length
                # 下面感觉没有任何作用
                max_length -= avg_max_length
                if max_length < 0:
                    raise AssertionError('Actual length has extended the maximum length.')
        
        return max_lengths
    
    def _create_position_ids_from_part(self, inputs, part, last_position):
        part_length = len(inputs)
        if 'positions' in part and part["positions"] >= 0:
            last_position = part['positions']
        
        if self.omask_token in part['text']:
            omask_id = self.tokenizer.convert_tokens_to_ids(self.omask_token)
            omask_index = [x for x in range(part_length) if inputs[x] == omask_id]
            omask_index = [0] + omask_index
            position_ids = []
            max_index = 0
            for start_id, end_id in zip(omask_index[:-1], omask_index[1:]):
                position_ids.extend(list(range(last_position, last_position + end_id - start_id)))
                max_index = max(max_index, end_id - start_id)
            
            difference = part_length - len(position_ids)
            position_ids.extend(range(last_position, last_position + difference))
            max_index = max(difference, max_index)
            last_position += max_index
        else:
            position_ids = list(range(last_position, last_position + part_length))
            last_position += part_length
        return position_ids, last_position

class UTCLoss(object):
    """
    adapted from paddlenlp
    """
    def __call__(self, logit, label):
        return self.forward(logit, label)
    
    def forward(self, logit, label):
        logit = (1.0 - 2.0 * label) * logit
        logit_neg = logit - label * 1e12
        logit_pos = logit - (1.0 - label) * 1e12
        zeros = torch.zeros_like(logit[..., :1], device=logit.device)
        logit_neg = torch.concat([logit_neg, zeros], axis=-1)
        logit_pos = torch.concat([logit_pos, zeros], axis=-1)
        label = torch.concat([label, zeros], axis=-1)
        logit_neg[label == -100] = -1e12
        logit_pos[label == -100] = -1e12
        neg_loss = torch.logsumexp(logit_neg, axis=-1)
        pos_loss = torch.logsumexp(logit_pos, axis=-1)
        loss = (neg_loss + pos_loss).mean()
        return loss

@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] =  True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    return_attention_mask: Optional[bool] = None
    default_model_input_names: List = (
        'input_ids',
        'token_type_ids',
        'special_tokens_mask',
        'offset_mapping',
        # 'position_ids'
    )

    def _convert_to_tensors(self, data):
        if self.return_tensors == 'pt':
            return torch.tensor(data)
        return data
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0]:
            if key in self.default_model_input_names:
                batch[key] = [b[key] for b in features]
        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            return_attention_mask=self.return_attention_mask
        )
        max_length = batch['input_ids'].shape[1]
        for key in features[0]:
            if key not in self.default_model_input_names:
                values = [b[key] for b in features if key in b]
                if len(values) < len(features):
                    continue

                if key == 'masked_positions':
                    new_values = []
                    for index, value in enumerate(values):
                        value = np.array(value) + index * max_length
                        new_values.extend(value.tolist())
                    values = new_values
                elif key == 'attention_mask':
                    new_values = np.zeros([len(values), max_length, max_length])
                    for index, value in enumerate(values):
                        length = len(value)
                        new_values[index][:length, :length] = value
                    values = new_values
                elif key == 'position_ids':
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_length - len(value))
                elif key in ('omask_positions'):
                    max_num_option = max([len(x) for x in values])
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_num_option - len(value))
                elif key == 'labels':
                    if isinstance(values[0], list):
                        max_num_label = max([len(x) for x in values])
                        for index, value in enumerate(values):
                            values[index] = value + [-100] * (max_num_label - len(value))
                elif key != 'cls_positions':
                    continue
                batch[key] = self._convert_to_tensors(values)
        
        return batch
