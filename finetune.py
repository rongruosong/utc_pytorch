# coding=utf-8
import math
import torch
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.strategies import DDPStrategy
from torchmetrics import MetricCollection, F1Score
from train_argparse import parse_args
from template import UTCTemplate
from transformers import BertTokenizer, ErnieConfig, PreTrainedTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from model import UTC
from trainer import Trainer
from utils import DataCollatorWithPadding

def create_data_loader(
    fabric: Fabric,
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    num_workers: int = 4,
    batch_size: int = 32,
    mode: str = 'train',
) -> DataLoader:
    dataset = load_dataset('json', data_files=data_path)
    dataset = dataset['train']
    utc_template = UTCTemplate(tokenizer, max_seq_len)
    if mode == 'train':
        dataset = dataset.shuffle()

    if fabric.local_rank > 0:
        fabric.print('Waiting for main process to perform the mapping')
        fabric.barrier()
    dataset = dataset.map(
        lambda example: utc_template(example), 
        batched=False,
        remove_columns=['text_a', 'text_b', 'question', 'choices']
    )
    if fabric.local_rank == 0:
        fabric.print('Loading results from main process')
        fabric.barrier()
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collator)

def main():
    args = parse_args()
    args.num_classes = 11
    args.ignore_index = -100
    
    # Initialize Fabric
    logger = TensorBoardLogger(args.log_dir)
    if args.strategy == 'ddp':
        strategy = DDPStrategy(find_unused_parameters=True)
    fabric = Fabric(accelerator='cuda', devices=args.devices, strategy=strategy, loggers=logger)
    # fabric.launch()
    fabric.seed_everything(args.seed + fabric.global_rank)
    fabric.print('world_size {}, global_rank {}'.format(fabric.world_size, fabric.global_rank))
    
    # Load and initialize model
    fabric.print('load model ...')
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    config = ErnieConfig.from_pretrained(args.config_path)
    model = UTC.from_pretrained(args.pretrained_model_path, config=config)
    # model = fabric.setup_module(model)

    # Load data
    fabric.print('load train data ...')
    train_loader = create_data_loader(
        fabric, 
        args.train_data_path, 
        tokenizer=tokenizer, 
        batch_size=args.train_batch_size, 
        max_seq_len=args.seq_length,
    )

    fabric.print('load val data ...')
    val_loader = create_data_loader(
        fabric, 
        args.test_data_path, 
        tokenizer=tokenizer, 
        batch_size=args.test_batch_size, 
        max_seq_len=args.seq_length,
        mode='val'
    )


    # Setup optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    estimated_stepping_batches = math.ceil(len(train_loader) / args.grad_accum_steps) * max(args.max_epochs, 1)
    
    num_warmup_steps = args.warmup * estimated_stepping_batches
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=estimated_stepping_batches,
    )
    
    fabric.print('total steps: {}'.format(estimated_stepping_batches))
    # create metircs
    micro_conf = {
        'task': args.task_type, 
        'num_labels': args.num_classes, 
        'num_classes': args.num_classes, 
        'average': 'micro', 
        'ignore_index': args.ignore_index
    }
    macro_conf = {
        'task': args.task_type, 
        'num_labels': args.num_classes, 
        'num_classes': args.num_classes, 
        'average': 'macro', 
        'ignore_index': args.ignore_index
    }
    train_metrics = MetricCollection(
        { 
            'micro_f1': F1Score(**micro_conf),
            'macro_f1': F1Score(**macro_conf)
        }
    )
    test_metrics = MetricCollection(
        { 
            'micro_f1': F1Score(**micro_conf),
            'macro_f1': F1Score(**macro_conf)
        }
    )
    # train
    fabric.print('begin to fit model')
    trainer = Trainer(
        args, 
        fabric, 
        optimizer=optimizer, 
        train_metrics=train_metrics, 
        test_metrics=test_metrics,
        scheduler=scheduler
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
     torch.set_float32_matmul_precision("high")
     main()