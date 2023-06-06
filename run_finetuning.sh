#!/bin/sh
CURRENT_DIR=`pwd`

# accelerator="gpu"
# strategy="ddp"
# devices=1

train_data_path=$CURRENT_DIR/data/train.txt
test_data_path=$CURRENT_DIR/data/dev.txt

config_path=/gemini/pretrain/config.json
vocab_path=/gemini/pretrain/

pretrained_model_path=/gemini/pretrain/pytorch_model.bin
checkpoint_dir=/gemini/checkpoint_dir
log_dir=/gemini/output

train_batch_size=2
test_batch_size=2
seq_length=512
learning_rate=1e-5

max_epochs=20

max_grad_norm=1
grad_accum_steps=1
eval_steps=10
logging_steps=10
save_checkpoint_steps=100

seed=42

lightning run model finetune.py \
    --train_data_path $train_data_path \
    --test_data_path $test_data_path \
    --config_path $config_path \
    --vocab_path $vocab_path \
    --pretrained_model_path $pretrained_model_path \
    --checkpoint_dir $checkpoint_dir \
    --log_dir $log_dir \
    --train_batch_size $train_batch_size \
    --test_batch_size $test_batch_size \
    --seq_length $seq_length \
    --learning_rate $learning_rate \
    --max_epochs $max_epochs \
    --max_grad_norm $max_grad_norm \
    --grad_accum_steps $grad_accum_steps \
    --eval_steps $eval_steps \
    --logging_steps $logging_steps \
    --save_checkpoint_steps $save_checkpoint_steps \
    --seed $seed