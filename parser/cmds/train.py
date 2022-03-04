# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from parser.cmds.cmd import CMD
from parser.helper.metric import Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch
import numpy as np
from parser.helper.util import *
from parser.helper.data_module import DataModule
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

import math

class Train(CMD):

    def __call__(self, args):

        self.args = args
        self.device = args.device

        dataset = DataModule(args)
        self.model = get_model(args.model, dataset)
        if hasattr(args, 'pretrained_model'):
            with open(args.pretrained_model, 'rb') as f:
                self.model.load_state_dict(torch.load(f))
        create_save_path(args)
        log = get_logger(args)
        self.optimizer = get_optimizer(args.optimizer, self.model)
        log.info("Create the model")
        log.info(f"{self.model}\n")
        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        log.info(self.optimizer)
        log.info(args)
        eval_loader = dataset.val_dataloader

        self.writer = SummaryWriter(args.save_dir)

        '''
        Training
        '''
        train_arg = args.train
        self.train_arg = train_arg

        # Arguments for validation
        eval_depth = self.args.test.eval_depth if hasattr(self.args.test, 'eval_depth') else False
        left_binarization = self.args.test.left_binarization if hasattr(self.args.test, 'left_binarization') else False
        right_binarization = self.args.test.right_binarization if hasattr(self.args.test, 'right_binarization') else False
        
        # Check total iteration
        self.iter = 0
        self.pf = []
        self.partition = False

        for epoch in range(1, train_arg.max_epoch + 1):
            '''
            Auto .to(self.device)
            '''

            # Warmup for epoch
            if hasattr(self.train_arg, 'warmup_epoch'):
                if epoch > self.train_arg.warmup_epoch:
                    self.partition = True

            # curriculum learning. Used in compound PCFG.
            if train_arg.curriculum:
                train_loader = dataset.train_dataloader(max_len=min(train_arg.start_len + epoch - 1, train_arg.max_len))
            else:
                train_loader = dataset.train_dataloader(max_len=train_arg.max_len)

            # print_depth = self.model.depth if hasattr(self.model, 'depth') or self.model.depth == 0 else 'Not estimate.'
            # log.info(f'GIL Depth: {print_depth}')

            train_loader_autodevice = DataPrefetcher(train_loader, device=self.device)
            eval_loader_autodevice = DataPrefetcher(eval_loader, device=self.device)
            start = datetime.now()
            self.train(train_loader_autodevice)
            log.info(f"Epoch {epoch} / {train_arg.max_epoch}:")


            dev_f1_metric, _, dev_ll, dev_left_metric, dev_right_metric = self.evaluate(eval_loader_autodevice,
                eval_depth=eval_depth, left_binarization=left_binarization, right_binarization=right_binarization)
            log.info(f"{'dev f1:':6}   {dev_f1_metric}")
            log.info(f"{'dev ll:':6}   {dev_ll}")
            # F1 score for each epoch
            self.writer.add_scalar('valid/F1', dev_f1_metric.sentence_uf1, epoch)
            self.writer.add_scalar('valid/F1_left', dev_left_metric.sentence_uf1, epoch)
            self.writer.add_scalar('valid/F1_right', dev_right_metric.sentence_uf1, epoch)
            self.writer.add_scalar('valid/Exact', dev_f1_metric.sentence_ex, epoch)
            self.writer.add_scalar('valid/Exact_left', dev_left_metric.sentence_ex, epoch)
            self.writer.add_scalar('valid/Exact_right', dev_right_metric.sentence_ex, epoch)
            # partition function distribution
            for i, pf in enumerate(self.pf_sum):
                self.writer.add_scalar('valid/partition_function', pf/dev_f1_metric.n, i)
            # F1 score for each depth
            for k, v in dev_f1_metric.sentence_uf1_d.items():
                self.writer.add_scalar('valid/f1_depth', v, k)
            # F1 score for each length
            # for k, v in dev_f1_metric.sentence_uf1_l.items():
            #     self.writer.add_scalar('valid/f1_length', v, k)
            for k, v in dev_f1_metric.sentence_ex_d.items():
                self.writer.add_scalar('valid/Ex_depth', v, k)
                
            # F1 score for each depth
            for k, v in dev_left_metric.sentence_uf1_d.items():
                self.writer.add_scalar('valid/f1_left_depth', v, k)
            # F1 score for each length
            # for k, v in dev_left_metric.sentence_uf1_l.items():
            #     self.writer.add_scalar('valid/f1_left_length', v, k)
            for k, v in dev_left_metric.sentence_ex_d.items():
                self.writer.add_scalar('valid/Ex_left_depth', v, k)

            # F1 score for each depth
            for k, v in dev_right_metric.sentence_uf1_d.items():
                self.writer.add_scalar('valid/f1_right_depth', v, k)
            # F1 score for each length
            # for k, v in dev_right_metric.sentence_uf1_l.items():
            #     self.writer.add_scalar('valid/f1_right_length', v, k)
            for k, v in dev_right_metric.sentence_ex_d.items():
                self.writer.add_scalar('valid/Ex_right_depth', v, k)

            # distribution of estimated span depth
            self.estimated_depth = dict(sorted(self.estimated_depth.items()))
            for k, v in self.estimated_depth.items():
                self.writer.add_scalar('valid/estimated_depth', v/dev_f1_metric.n, k)

            t = datetime.now() - start

            # save the model if it is the best so far
            if dev_ll > best_metric:
                best_metric = dev_ll 
                best_e = epoch
                torch.save(
                   obj=self.model.state_dict(),
                   f = args.save_dir + "/best.pt"
                )
                log.info(f"{t}s elapsed (saved)\n")
            else:
                log.info(f"{t}s elapsed\n")

            total_time += t
            if train_arg.patience > 0 and epoch - best_e >= train_arg.patience:
                break
        
        self.writer.flush()
        self.writer.close()
