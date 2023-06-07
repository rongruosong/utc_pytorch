# coding=utf-8

import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='information extraction')

    # task type
    parser.add_argument("--task_type",choices=["multiclass", "multilabel"],
                        default="multiclass",
                        help="task type.")
    # env options
    parser.add_argument("--devices", type=int, default=1,
                        help='the number of GPU')
    parser.add_argument("--strategy", type=str, default='ddp',
                        help='parallel strategy')
    

    # dataset path options
    parser.add_argument("--train_data_path", type=str,
                        help='path of the train dataset')
    parser.add_argument("--test_data_path", type=str,
                        help='path of the test dataset')
    
    # bert config options
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path of the config file.")

    # tokenizer options
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    
    # model options
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="Path of the output model.")

    # data options
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Batch size of train dataset.")
    parser.add_argument("--test_batch_size", type=int, default=32,
                        help="Batch size of test dataset.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    
    # optimization options
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warmup ratio value.")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"],
                        default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                                "constant", "constant_with_warmup"],
                        default="linear", help="Scheduler type.")

    # trainer options
    parser.add_argument("--max_epochs", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Number of steps.")
    parser.add_argument("--max_grad_norm", default=1, type=float,
                        help="gradient clip norm val.")
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help="How often within one training epoch to check the validation set")
    parser.add_argument('--log_dir', type=str, default='/gemini/output')

    # report step
    parser.add_argument("--logging_steps", type=int, default=200,
                        help="Specific steps to log.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=200,
                        help="Specific steps to checkpoint save.")

    # evn options
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    args = parser.parse_args()
    return args