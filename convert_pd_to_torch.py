# coding=utf-8
import os
import argparse

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.transformers import (ErnieForPretraining, ErniePretrainedModel,
                                    ErnieTokenizer, AutoTokenizer)
from paddlenlp.transformers import UTC as PDUTC
from transformers import BertConfig, BertTokenizer, BertTokenizerFast
import torch
from model import UTC

def change_paddle_key():
    paddle_state_dict = {}

    # embedding
    paddle_state_dict['ernie.embeddings.word_embeddings.weight'] = 'ernie.embeddings.word_embeddings.weight'
    paddle_state_dict['ernie.embeddings.position_embeddings.weight'] = 'ernie.embeddings.position_embeddings.weight'
    paddle_state_dict['ernie.embeddings.token_type_embeddings.weight'] = 'ernie.embeddings.token_type_embeddings.weight'
    paddle_state_dict['ernie.embeddings.LayerNorm.weight'] = 'ernie.embeddings.layer_norm.weight'
    paddle_state_dict['ernie.embeddings.LayerNorm.bias'] = 'ernie.embeddings.layer_norm.bias'

    # encoder
    for i in range(12):
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.query.weight'.format(i)] = 'ernie.encoder.layers.{}.self_attn.q_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.query.bias'.format(i)] = 'ernie.encoder.layers.{}.self_attn.q_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.key.weight'.format(i)] = 'ernie.encoder.layers.{}.self_attn.k_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.key.bias'.format(i)] = 'ernie.encoder.layers.{}.self_attn.k_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.value.weight'.format(i)] = 'ernie.encoder.layers.{}.self_attn.v_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.value.bias'.format(i)] = 'ernie.encoder.layers.{}.self_attn.v_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.dense.weight'.format(i)] = 'ernie.encoder.layers.{}.self_attn.out_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.dense.bias'.format(i)] = 'ernie.encoder.layers.{}.self_attn.out_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.LayerNorm.weight'.format(i)] = 'ernie.encoder.layers.{}.norm1.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.LayerNorm.bias'.format(i)] = 'ernie.encoder.layers.{}.norm1.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.intermediate.dense.weight'.format(i)] = 'ernie.encoder.layers.{}.linear1.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.intermediate.dense.bias'.format(i)] = 'ernie.encoder.layers.{}.linear1.bias'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.dense.weight'.format(i)] = 'ernie.encoder.layers.{}.linear2.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.dense.bias'.format(i)] = 'ernie.encoder.layers.{}.linear2.bias'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.LayerNorm.weight'.format(i)] = 'ernie.encoder.layers.{}.norm2.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.LayerNorm.bias'.format(i)] = 'ernie.encoder.layers.{}.norm2.bias'.format(i) 

    paddle_state_dict['ernie.pooler.dense.weight'] = 'ernie.pooler.dense.weight'
    paddle_state_dict['ernie.pooler.dense.bias'] = 'ernie.pooler.dense.bias'
    paddle_state_dict['linear_q.weight'] = 'linear_q.weight'
    paddle_state_dict['linear_q.bias'] = 'linear_q.bias'
    paddle_state_dict['linear_k.weight'] = 'linear_k.weight'
    paddle_state_dict['linear_k.bias'] = 'linear_k.bias'

    return paddle_state_dict

def convert(model, pd_model_weight_path, save_path):
    # 加载paddle参数
    paddle_key_params = paddle.load(pd_model_weight_path)

    paddle_state_dict = change_paddle_key()
    state_dict = model.state_dict()
    for key in state_dict.keys():
        
        if key in paddle_state_dict.keys():
            param = paddle_key_params[paddle_state_dict[key]]
            if 'weight' in key and 'LayerNorm' not in key and 'embeddings' not in key and 'decoder' not in key:
                param = param.transpose((1, 0))
            state_dict[key] = torch.from_numpy(param.numpy())
        else:
            print(key)
    model.load_state_dict(state_dict, strict=False)
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    pd_model_weight_path = '/root/autodl-nas/uie-base/model_state.pdparams'
    config_dir = '/gemini/pretrain/model_config.json'
    model_path = '/gemini/pretrain/pytorch_model.bin'

    config = BertConfig.from_json_file(config_dir)
    model = UIETorch.from_pretrained(model_path, config=config)

    save_path = '/root/autodl-nas/uie-base/pytorch_model.bin'
    convert(model, pd_model_weight_path, save_path)

    # 验证参数转换后， torch的结果是否与paddle保持一致
    tokenizer = BertTokenizerFast.from_pretrained('/gemini/pretrain/')
    example = {"content": "房间很好，房间很大，服务周到，性价比较高，价格很贵", "result_list": [{"text": "周到", "start": 12, "end": 14}], "prompt": "服务的观点词"}
    
    encoded_inputs = tokenizer.encode_plus(text=example["prompt"],
                                                        text_pair=example["content"],
                                                        padding=False,
                                                        truncation=True,
                                                        return_offsets_mapping=True)
    tokenized_output = [
        encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
        encoded_inputs["attention_mask"]
    ]

    # torch
    model.eval()
    print(model(input_ids=tokenized_input[0], token_type_ids=tokenized_input[1], att_mask=tokenized_input[2]))


    # paddle
    tokenizer = ErnieTokenizer.from_pretrained('uie-base')

    tokenized_input = [paddle.to_tensor(np.expand_dims(x, 0)) for x in tokenized_output]
    model = UIE.from_pretrained('uie-base')
    model.eval()

    print(model(input_ids=tokenized_input[0],
        token_type_ids=tokenized_input[1],
        attention_mask=tokenized_input[2]))