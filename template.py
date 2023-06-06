# coding=utf-8
"""
adapted from paddlenlp
"""
from typing import Optional, List, Dict, Any
import os
import traceback
from utils import MLMTokenizerWrap
from transformers import PreTrainedTokenizer
import numpy as np

DEFAULT_MAX_OPTIONS = 10
class Template:
    input_feature_names = ["do_truncate", "token_types", "positions"]
    omask_token = '[O-MASK]'
    opt_token = '[OPT]'
    def __init__(self, prompt: str, tokenizer: PreTrainedTokenizer, max_length: int, **kwargs):
        super(Template, self).__init__()
        for key, value in kwargs:
            setattr(self, key, valus)
        self.tokenizer = tokenizer
        self.prompt_tokenizer = MLMTokenizerWrap(tokenizer, max_length)
        self.set_prompt(prompt)
    
    @property
    def prompt(self):
        return self._prompt

    def set_prompt(self, prompt: str):
        if prompt is not None:
            if isinstance(prompt, str):
                self._prompt = self.parse_template_string(prompt)
            else:
                self._prompt = prompt
        
        self.do_truncate = self.create_truncation_sequence_from_prompt()
        self.example_keys = self.create_example_keys_from_prompt()
        self.token_types = self.create_token_type_sequence_from_prompt()
        self.positions = self.create_position_sequence_from_prompt()
        self._check_omask_token()
    
    def create_truncation_sequence_from_prompt(self, prompt: Optional[List[Dict[str, Any]]] = None) -> List[bool]:
        prompt = self._prompt if prompt is None else prompt
        do_truncate = []
        for part in prompt:
            if 'truncate' in part:
                do_truncate.append(part['truncate'])
            elif 'text' in part:
                do_truncate.append(True)
            else:
                do_truncate.append(False)
        return do_truncate

    def create_example_keys_from_prompt(self):
        example_keys = set()
        for part in self._prompt:
            if 'text' in part:
                example_keys.add(part['text'])
            if 'options' in part and isinstance(part['options'], list):
                example_keys.update(set(part['options']))
        
        if len(example_keys) == 0:
            raise ValueError('No `text` keyword in template: "{}", please check it again.'.format(self.prompt))

        return example_keys

    def create_token_type_sequence_from_prompt(self, prompt: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        prompt = self._prompt if prompt is None else prompt
        token_type = []
        last_token_type = 0
        for part in prompt:
            if 'token_type' in part:
                last_token_type = part["token_type"]
            token_type.append(last_token_type)
        return token_type

    def create_position_sequence_from_prompt(self, prompt: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        prompt = self._prompt if prompt is None else prompt
        position_ids = []
        for part in prompt:
            if 'position' in part:
                position_ids.append(part['position'])
            else:
                position_ids.append(-1)
        return position_ids

    def _check_omask_token(self):
        prompt = self._prompt
        for part in prompt:
            if 'add_omask' in part:
                if self.omask_token not in self.tokenizer.additional_special_tokens:
                    self.tokenizer.add_special_tokens({"additional_special_tokens": [self.omask_token]})

    def build_inputs_with_prompt(
        self, example: Dict[str, Any], 
        prompt: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        根据prompt和example创建相关文本输入
        for example, if the example:
        {'text_a': '糖尿病蜜月期有永久性的吗','text_b': '','question': '',
         'choices': ['病情诊断', '治疗方案', '病因分析', '其他'], 'labels': [3]}
        then this function returns
        [{'text': '[O-MASK]病情诊断[O-MASK]治疗方案[O-MASK]病因分析[O-MASK]其他', 'do_truncate': False, 'token_types': 1, 'positions': 0}, {'text': '[SEP]', 'do_truncate': False, 'token_types': 0, 'positions': 0}, {'text': '糖尿病蜜月期有永久性的吗', 'do_truncate': True, 'token_types': 0, 'positions': -1}, {'text': '[SEP]', 'do_truncate': False, 'token_types': 1, 'positions': -1}, {'text': '', 'do_truncate': True, 'token_types': 1, 'positions': -1}]
        """
        inputs = []
        prompt = self._prompt if prompt is None else prompt
        
        for index, part in enumerate(prompt):
            if 'text' in part:
                if part['text'] not in example:
                    raise ValueError(
                        "Unexpected value in template.Can not find keyword {} in example:{}".format(part['text'], example)
                    )
                inputs.append(example[part['text']])
            elif 'mask' in part:
                if 'length' not in part:
                    part['length'] = 1
                input.append(self.tokenizer.mask_token * part['length'])
            elif 'sep' in part:
                inputs.append(self.tokenizer.sep_token)
            elif 'hard' in part:
                inputs.append(part['hard'])
            elif 'options' in part:
                if not isinstance(part['options'], list):
                    if part['options'] not in example:
                        raise ValueError(
                            "Unexpected value in template.Can not find keyword {} in example:{}".format(part['text'], example)
                        )
                    labels = example[part['options']]
                    labels = [labels] if isinstance(labels, str) else labels
                else:
                    labels = part['options']
                if 'add_prompt' in part:
                    opt_prompt = part['add_prompt']
                    lables = [opt_prompt.replace(self.opt_token, x) for x in labels]
                if 'add_omask' in part:
                    labels = [self.omask_token + x for x in labels]
                inputs.append("".join(labels))
            else:
                inputs.append(part)
            
            if 'add_space' in part:
                inputs[-1] = ' ' + inputs[-1]
        return inputs

    def encode(self, example: Dict[str, Any]):
        input_text = self.build_inputs_with_prompt(example)
        input_names, input_values = ['text'], [input_text]
        for name in self.input_feature_names:
            input_names.append(name)
            input_values.append(getattr(self, name, None))
        
        inputs = []
        for value in list(zip(*input_values)):
            inputs.append(dict(zip(input_names, value)))
        input_dict = self.prompt_tokenizer(inputs)

        return input_dict
        
    def __call__(self, example: Dict[str, Any]):
        return self.encode(example)

    @staticmethod
    def parse_template_string(prompt: str, left_token: Optional[str]='{', right_token: Optional[str] = '}'):
        """
        parse the prompt string
        """
        parsed = []
        index = 0
        while index < len(prompt):
            part = {"add_space": ' '} if prompt[index] == ' ' else {}
            while index < len(prompt) and prompt[index] == ' ':
                index += 1
            if index == len(prompt): 
                break
            
            if prompt[index] == left_token:
                left_index = index
                while index < len(prompt):
                    if prompt[index] == right_token:
                        break
                    index += 1

                try:
                    part_dict = eval(prompt[left_index: index + 1])
                    if isinstance(part_dict, set):
                        part_dict = {k: None for k in part_dict}
                    part.update(part_dict)
                except SyntaxError:
                    print(traceback.format_exc())
                    exit()
                index += 1
            else:
                # parse simplified discrete prompts
                left_index = index
                while index < len(prompt) and prompt[index] != left_token:
                    index += 1
                part['hard'] = prompt[left_index: index].rstrip(" ")
            
            if 'options' in part:
                if os.path.isfile(part['options']):
                    with open(part['options'], 'r') as fp:
                        labels = [x.strip() for x in fp]
                    part['options'] = labels
                    part['length'] = len(labels)
                elif 'length' not in 'options':
                    part['length'] = DEFAULT_MAX_OPTIONS
            
            if 'position' in part:
                assert part['position'] >= 0
            if 'truncate' in part:
                assert part['truncate'] in [True, False]
            
            parsed.append(part)
        
        return parsed

class UTCTemplate(Template):
    """
    Template for Unified Tag Classification.
    """

    template_special_tokens = ["text", "hard", "sep", "cls", "options"]

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, prompt: str = None):
        prompt = (
            (
                "{'options': 'choices', 'add_omask': True, 'position': 0, 'token_type': 1}"
                "{'sep': None, 'token_type': 0, 'position': 0}{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}"
            )
            if prompt is None
            else prompt
        )
        super(UTCTemplate, self).__init__(prompt, tokenizer, max_length)
        self.max_position_id = self.tokenizer.model_max_length - 1
        self.max_length = max_length
        if not self._has_options():
            raise ValueError(
                "Expected `options` and `add_omask` are in defined prompt, but got {}".format(self.prompt)
            )

    def _has_options(self):
        for part in self.prompt:
            if "options" in part and "add_omask" in part:
                return True
        return False

    def build_inputs_with_prompt(
        self, example: Dict[str, Any], prompt: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        inputs = super(UTCTemplate, self).build_inputs_with_prompt(example, prompt)
        for index, part in enumerate(inputs):
            if "cls" in part:
                inputs[index] = self.tokenizer.cls_token
        return inputs

    def encode(self, example: Dict[str, Any], use_mask: bool = False):
        input_dict = super(UTCTemplate, self).encode(example)

        # Set OMASK and MASK positions and labels for options.
        omask_token_id = self.tokenizer.convert_tokens_to_ids("[O-MASK]")
        input_dict['omask_positions'] = (
            np.where(np.array(input_dict['input_ids']) == omask_token_id)[0].squeeze().tolist()
        )

        sep_positions = (
            np.where(np.array(input_dict['input_ids']) == self.tokenizer.sep_token_id)[0].squeeze().tolist()
        )
        input_dict['cls_positions'] = sep_positions[0]
        if 'labels' in example:
            if not isinstance(example["labels"], list):
                example['labels'] = [example['labels']]
            one_hots = np.zeros(len(example['choices']), dtype="float32")
            for x in example['labels']:
                one_hots[x] = 1
            example['labels'] = one_hots.tolist()

        # Limit the maximum position ids.
        position_ids = np.array(input_dict['position_ids'])
        position_ids[position_ids > self.max_position_id] = self.max_position_id
        input_dict['position_ids'] = position_ids.tolist()

        return input_dict

    def create_prompt_parameters(self):
        return None

    def process_batch(self, input_dict):
        return input_dict