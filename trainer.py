# coding=utf-8
import os
from typing import Any, Optional, Union, List, Dict, Callable, Iterable
import torch
import torch.nn as nn
from torch.optim import Optimizer
import argparse
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy, DeepSpeedStrategy
from lightning_utilities import apply_to_collection
from transformers import AdamW, get_linear_schedule_with_warmup

from torchmetrics import MeanMetric, Metric, MetricCollection, F1Score

class Trainer:
    def __init__(
        self,
        args: argparse.Namespace,
        fabric: Fabric,
        optimizer: Optimizer,
        scheduler: Optional[Callable] = None
    ) -> None:
        self.fabric = fabric

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.max_epochs = args.max_epochs
        self.max_steps = args.max_steps
        self.max_grad_norm = args.max_grad_norm
        self.grad_accum_steps = args.grad_accum_steps

        self.logging_steps = args.logging_steps
        # self.eval_logging_steps = args.eval_logging_steps
        self.eval_steps = args.eval_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.checkpoint_dir = args.checkpoint_dir

        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index

        self.current_step = 0
        self.global_step = 0
        self.current_epoch = 0

        self.train_loss = self.fabric.to_device(MeanMetric())
        self.train_metrics = self.fabric.to_device(
            MetricCollection(
                { 
                    'micro_f1': F1Score(task='multilabel', num_labels=self.num_classes, average='micro', ignore_index=self.ignore_index),
                    'macro_f1': F1Score(task='multilabel', num_labels=self.num_classes, average='macro', ignore_index=self.ignore_index)
                }
            )
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input_ids, token_type_ids, position_ids, attention_mask, omask_positions, cls_positions, labels = batch
        output = self.model(
            **batch
        )
        # output.update({'labels': batch['labels']})
        return output

    def train_step(self, batch: Dict[str, torch.Tensor]):
        output = self.forward(batch)
       
        metric_output = apply_to_collection(output, torch.Tensor, lambda x: x.detach())
        self.train_loss.update(metric_output['loss'])
        self.train_metrics(metric_output['option_logits'], batch['labels'])
        return output['loss']

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        for i, batch in enumerate(train_loader):
            # Accumulate gradient 8 batches at a time
            is_accumulating = (self.global_step + 1) % self.grad_accum_steps != 0

            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                loss = self.train_step(batch)
                if self.grad_accum_steps > 1:
                    loss = loss / self.grad_accum_steps
                self.fabric.backward(loss)

            if not is_accumulating:
                # Step the optimizer after accumulation phase is over
                if self.max_grad_norm is not None and self.max_grad_norm > 0 and \
                    not isinstance(self.fabric.strategy, DeepSpeedStrategy) :
                    self.fabric.clip_gradients(self.model, optimizer=self.optimizer, max_norm=self.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.fabric.log('lr', self.scheduler.get_last_lr()[0], step=self.current_step)
                    self.scheduler.step()
                else:
                    self.fabric.log('lr',self.optimizer.param_groups[0]['lr'], step=self.current_step)
                    
                self.optimizer.zero_grad()
                self.current_step += 1
            
                if self.current_step % self.logging_steps == 0 and self.fabric.is_global_zero:
                    self.log_info(self.train_loss, self.train_metrics, 'train')
                
                if self.current_step % self.eval_steps == 0 and self.fabric.is_global_zero:
                    self.eval(val_loader)
                
                if self.current_step % self.save_checkpoint_steps == 0:
                    state = {
                        'model': self.model,
                        'current_step': self.current_step,
                        'current_epoch': self.current_epoch
                    }
                    self.save(state)
                    
            self.global_step += 1       

    def eval(self, val_loader: DataLoader):
        torch.set_grad_enabled(False)
        test_loss = self.fabric.to_device(MeanMetric())
        test_metrics = self.fabric.to_device(
            MetricCollection(
                { 
                    'micro_f1': F1Score(task='multilabel', num_labels=self.num_classes, average='micro', ignore_index=self.ignore_index),
                    'macro_f1': F1Score(task='multilabel', num_labels=self.num_classes, average='macro', ignore_index=self.ignore_index)
                }
            )
        )
        for i, batch in enumerate(val_loader):
            output = self.forward(batch)
            output = apply_to_collection(output, torch.Tensor, lambda x: x.detach())
            test_loss.update(output['loss'])
            test_metrics.update(output['option_logits'], batch['labels'])
        self.log_info(test_loss, test_metrics, 'eval')
        torch.set_grad_enabled(True)
    
    def fit(
        self, 
        model: nn.Module, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        train_loader = self.fabric.setup_dataloaders(train_loader)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=False)
        self.model, self.optimizer = self.fabric.setup(model, self.optimizer)

        self.model.train()
        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch
            self.train_epoch(train_loader, val_loader)
    
    def test(
        self,
        model: nn.Module,
        val_loader: DataLoader
    ):
        val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=False)
        self.model = self.fabric.setup(model)
        self.eval(val_loader)
    
    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def log_info(self, loss_metric: MeanMetric, f1_metrics: MetricCollection, mode: str = 'train'):
        loss = loss_metric.compute()
        metrics = f1_metrics.compute()
        log_metrics = {
            f'{mode}_loss': loss,
            f'{mode}_micro_f1': metrics['micro_f1'], 
            f'{mode}_macro_f1': metrics['macro_f1'], 
        }
        loss_metric.reset()
        f1_metrics.reset()
        self.fabric.log_dict(log_metrics, self.current_step)
        log_metrics = apply_to_collection(log_metrics, torch.Tensor, lambda x: x.item())
        self.fabric.print('{} steps: {}, loss: {}, micro_f1: {}, macro_f1: {}'.format(mode, self.current_step, *log_metrics.values()))
    
    def save(self, state):
        self.fabric.save(os.path.join(self.checkpoint_dir, f"step-{self.current_step:04d}.ckpt"), state)
