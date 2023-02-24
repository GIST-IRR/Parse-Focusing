# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm


class Trainer:
    model: nn.Module
    optimizer: Optimizer

    _train_batch_pre_hook_list = []
    _train_batch_post_hook_list = []
    _eval_batch_pre_hook_list = []
    _eval_batch_post_hook_list = []

    def __init__(self, args, model, optimizer) -> None:
        self.args = args
        self.model = model
        self.optimizer = optimizer

    def _execute_hooks(self, hook_list):
        for hook, args, kwargs in hook_list:
            hook(*args, **kwargs)
    # Train hooks
    def train_batch_pre_hook(self):
        self._execute_hooks(self._train_batch_pre_hook_list)

    def train_batch_post_hook(self):
        self._execute_hooks(self._train_batch_post_hook_list)
    # Evaluation hooks
    def eval_batch_pre_hook(self):
        self._execute_hooks(self._eval_batch_pre_hook_list)
    
    def eval_batch_post_hook(self):
        self._execute_hooks(self._eval_batch_post_hook_list)

    def register_train_batch_pre_hook(self, hook, *args, **kwargs):
        self._register_hook(
            self._train_batch_pre_hook_list, hook, *args, **kwargs
        )

    def register_train_batch_post_hook(self, hook, *args, **kwargs):
        self._register_hook(
            self._train_batch_post_hook_list, hook, *args, **kwargs
        )

    def register_eval_batch_pre_hook(self, hook, *args, **kwargs):
        self._register_hook(
            self._eval_batch_pre_hook_list, hook, *args, **kwargs
        )

    def register_eval_batch_post_hook(self, hook, *args, **kwargs):
        self._register_hook(
            self._eval_batch_post_hook_list, hook, *args, **kwargs
        )

    def _register_hook(self, hook_list, hook, *args, **kwargs):
        assert isinstance(hook, function), \
            f'hook must be a function, but got {type(hook)}'
        hook_list.append((hook, args, kwargs))

    # Training
    def train_batch(self, x):
        self.optimizer.zero_grad()
        loss = self.model(x)
        loss.backward()
        self.total_loss += loss.item()
        self.optimizer.step()
        return None

    def train_epoch(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)), position=0, leave=True)

        for x, _ in t:
            self.train_batch_pre_hook()
            results = self.train_batch(x)
            self.train_batch_post_hook()
            self.iter += 1
        return results

    def train(self):
        #TODO: Dataloader
        for epoch in range(self.args.epochs):
            # Training
            self.train_epoch()
            # Evaluation
            self.evaluate_epoch()

            # If best,
            # Save model

    # Evaluation
    @torch.no_grad()
    def evaluate_batch(self, x):
        pred = self.model.predict(x)
        return None

    @torch.no_grad()
    def evaluate_epoch(self, loader):
        self.model.eval()

        #TODO: MetricDict
        t = tqdm(loader, total=int(len(loader)), position=0, leave=True)

        for x, y in t:
            self.eval_batch_pre_hook()
            results = self.evaluate_batch(x)
            self.eval_batch_post_hook()
        return results

    def evaluate(self):
        self.evaluate_epoch()
        #TODO: Logging