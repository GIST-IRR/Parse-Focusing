# -*- coding: utf-8 -*-
from parser.helper.conflict_detector import ConflictDetector

from datetime import datetime, timedelta
from parser.cmds.cmd import CMD
from parser.helper.metric import LikelihoodMetric, Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch
import numpy as np
# from parser.helper.util import *
from parser.helper.data_module import DataModule

from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

from pathlib import Path
import math
from utils import tensor_to_heatmap

import torch_support.reproducibility as reproducibility
from torch_support.train_support import (
    get_logger
)
from torch_support.load_model import (
    get_model_args,
    get_optimizer_args
)

class Train(CMD):

    def __call__(self, args):

        self.args = args
        self.device = args.device

        # Load pretrained model
        start_epoch = 1
        if hasattr(args, 'pretrained_model'):
            checkpoint = reproducibility.load(args.pretrained_model)
            # Load meta data
            start_epoch = checkpoint['epoch'] + 1
            # Load random state
            worker_init_fn = checkpoint['worker_init_fn']
            generator = checkpoint['generator']
        else:
            if hasattr(args, 'seed'):
                worker_init_fn, generator = \
                    reproducibility.fix_seed(args.seed)

        # Load dataset
        dataset = DataModule(
            args,
            generator=generator,
            worker_init_fn=worker_init_fn
        )
        # Update vocab size
        args.model.update({"V": len(dataset.word_vocab)})

        # Setup model
        self.model = get_model_args(args.model, self.device)
        # Load pretrained model
        if hasattr(args, 'pretrained_model'):
            self.model.load_state_dict(checkpoint['model'])

        # Setup optimizer
        self.optimizer = get_optimizer_args(
            args.optimizer, self.model.withoutTerm_parameters()
        )
        self.term_optimizer = get_optimizer_args(
            args.term_optimizer, self.model.terms.parameters()
        )
        # Load pretrained optimizer
        if hasattr(args, 'pretrained_model'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Setup logger
        self.conflict_detector = ConflictDetector(
            dataset.train_dataset, self.model, self.optimizer
        )

        if hasattr(args, 'pretrained_terms'):
            # with open(args.pretrained_terms, 'rb') as f:
            #     checkpoint = torch.load(f, map_location=self.device)
            with open(args.pretrained_terms, 'rb') as f:
                terms_checkpoint = torch.load(f, map_location=self.device)
            # with open(args.pretrained_nonterms, 'rb') as f:
            #     nonterms_checkpoint = torch.load(f, map_location=self.device)

            # Load pretrained terms
            # model_dict = {
            #     '.'.join(k.split('.')[1:]): v
            #     for k, v in checkpoint['model'].items()
            #     # if ('terms.' in k and 'nonterms.' not in k) or 'enc_' in k
            #     if 'nonterms.' in k or 'enc_' in k
            # }
            terms_model_dict = {
                '.'.join(k.split('.')[1:]): v
                for k, v in terms_checkpoint['model'].items()
                if ('terms.' in k and 'nonterms.' not in k) or 'enc_' in k
                # if 'nonterms.' in k or 'enc_' in k
            }
            # nonterms_model_dict = {
            #     '.'.join(k.split('.')[1:]): v
            #     for k, v in nonterms_checkpoint['model'].items()
            #     # if ('terms.' in k and 'nonterms.' not in k) or 'enc_' in k
            #     if 'nonterms.' in k or 'enc_' in k
            # }

            # self.model.nonterms.load_state_dict(nonterms_model_dict)
            # for param in self.model.nonterms.parameters():
            #     param.requires_grad_(False)

            self.model.terms.load_state_dict(terms_model_dict)
            for param in self.model.terms.parameters():
                param.requires_grad_(False)

            # for name, param in self.model.named_parameters():
            #     if 'enc_' in name:
            #         param.requires_grad_(False)

            if hasattr(self.model, 'enc'):
                for param in self.model.enc.parameters():
                    param.requires_grad_(False)

        # Load word embeddings
        # self.word_vectors = gensim.models.KeyedVectors.load('word2vec-ptb-std.wordvectors')
        # self.model.word_emb = nn.Parameter(torch.tensor(self.word_vectors.vectors, device=args.device))
        # self.model.term_mlp2.weight = nn.Parameter(torch.tensor(self.word_vectors.vectors, device=args.device))

        # weights = torch.tensor(self.word_vectors.vectors, device=args.device)
        # self.model.word_emb = nn.Embedding.from_pretrained(weights)
        # self.model.word_emb.requires_grad_(False)

        # Load term embeddings
        # with open('word2vec-ptb-std.means', 'rb') as f:
        #     self.model.term_emb = nn.Parameter(torch.tensor(pickle.load(f), device=args.device))

        # Setup logger
        log = get_logger(args)
        if not hasattr(args, 'seed'):
            log.info(f'seed: {torch.initial_seed()}')
        log.info("Create the model")
        log.info(f"{self.model}\n")
        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        log.info(self.optimizer)
        log.info(args)
        eval_loader = dataset.val_dataloader

        # Setup tensorboard writer
        self.writer = SummaryWriter(args.save_dir)

        '''
        Training
        '''
        train_arg = getattr(args, "train")
        test_arg = getattr(args, "test")
        self.train_arg = train_arg
        self.test_arg = test_arg

        # Arguments for validation
        eval_depth = getattr(test_arg, "eval_depth", False)
        left_binarization = getattr(test_arg, "left_binarization", False)
        right_binarization = getattr(test_arg, "right_binarization", False)
        
        # iteration setup
        self.num_batch = len(
            dataset.train_dataloader(max_len=train_arg.max_len)
        )
        if hasattr(train_arg, "total_iter"):
            train_arg.max_epoch = math.ceil(
                train_arg.total_iter / self.num_batch
            )
            log.info(
                f'num of batch: {self.num_batch}, max epoch: {train_arg.max_epoch}'
            )
        train_arg.total_iter = train_arg.max_epoch * self.num_batch
        log.info(f'total iter: {train_arg.total_iter}')

        if getattr(train_arg, "dambda_warmup", False):
            train_arg.warmup_iter = int(train_arg.total_iter * train_arg.dambda_warmup)
            train_arg.warmup_start = int(train_arg.total_iter * (train_arg.dambda_warmup - 0.1))
            train_arg.warmup_end = int(train_arg.total_iter * (train_arg.dambda_warmup + 0.1))
            log.info(f'warmup start: {train_arg.warmup_start}, middle: {train_arg.warmup_iter}, end: {train_arg.warmup_end}')

        # Check total iteration
        self.iter = (start_epoch - 1) * self.num_batch
        self.pf = []
        self.partition = False
        self.total_loss = 0
        self.total_len = 0
        self.total_metrics = {}
        self.dambda = 1
        self.step = 1

        for epoch in range(start_epoch, train_arg.max_epoch + 1):
            '''
            Auto .to(self.device)
            '''

            # Warmup for epoch
            if hasattr(self.train_arg, 'warmup_epoch'):
                if epoch > self.train_arg.warmup_epoch:
                    self.partition = True

            # partition switch for each epoch
            # self.writer.add_scalar(
            #     'train/norm_switch', 1 if self.partition else 0, epoch
            # )

            # curriculum learning. Used in compound PCFG.
            if train_arg.curriculum:
                self.max_len = min(
                    train_arg.start_len + epoch - 1, train_arg.max_len
                )
                self.min_len = train_arg.min_len
            else:
                self.max_len = train_arg.max_len
                self.min_len = train_arg.min_len

            train_loader = dataset.train_dataloader(
                max_len=self.max_len, min_len=self.min_len
            )
            # if epoch == 1:
            self.num_batch = len(train_loader)
            self.total_iter = self.num_batch * train_arg.max_epoch

            train_loader_autodevice = DataPrefetcher(
                train_loader, device=self.device
            )
            eval_loader_autodevice = DataPrefetcher(
                eval_loader, device=self.device
            )
            start = datetime.now()
            self.train(train_loader_autodevice)

            # Visualization
            heatmap_dir = Path(self.args.save_dir) / 'heatmap'
            for k in self.total_metrics.keys():
                mp.Process(
                    target=tensor_to_heatmap,
                    args=(self.model.metrics[k], ),
                    kwargs={
                        "dirname": heatmap_dir,
                        "filename": f'{k}_{self.iter}.png'
                    })
                # tensor_to_heatmap(
                #     self.model.metrics[k],
                #     dirname=heatmap_dir,
                #     filename=f'{k}_{self.iter}.png'
                # )
            log.info(f"Epoch {epoch} / {train_arg.max_epoch}:")

            # Evaluation
            dev_f1_metric, _, dev_ll, dev_left_metric, dev_right_metric = \
                self.evaluate(
                    eval_loader_autodevice,
                    decode_type=args.test.decode,
                    eval_depth=eval_depth,
                    left_binarization=left_binarization,
                    right_binarization=right_binarization
                )
            log.info(f"{'dev f1:':6}   {dev_f1_metric}")
            log.info(f"{'dev ll:':6}   {dev_ll}")

            # F1 score for each epoch
            tag = "valid"
            metric_list = {
                "Likelihood": dev_ll.score,
                "F1": dev_f1_metric.sentence_uf1,
                "F1_left": dev_left_metric.sentence_uf1,
                "F1_right": dev_right_metric.sentence_uf1,
                "Exact": dev_f1_metric.sentence_ex,
                "Exact_left": dev_left_metric.sentence_ex,
                "Exact_right": dev_right_metric.sentence_ex,
            }
            for k, v in metric_list.items():
                self.writer.add_scalar(f"{tag}/{k}", v, epoch)

            # partition function distribution
            for i, pf in enumerate(self.pf_sum):
                self.writer.add_scalar(f'valid/marginal_{self.model.mode}', pf/dev_f1_metric.n, i)

            metric_dict = {
                "f1_length": dev_f1_metric.sentence_uf1_l,
                "Ex_length": dev_f1_metric.sentence_ex_l,
                "f1_left_length": dev_left_metric.sentence_uf1_l,
                "Ex_left_length": dev_left_metric.sentence_ex_l,
                "f1_right_length": dev_right_metric.sentence_uf1_l,
                "Ex_right_length": dev_right_metric.sentence_ex_l,
                # "f1_depth": dev_f1_metric.sentence_uf1_d,
                # "Ex_depth": dev_f1_metric.sentence_ex_d,
                # "f1_left_depth": dev_left_metric.sentence_uf1_d,
                # "Ex_left_depth": dev_left_metric.sentence_ex_d,
                # "f1_right_depth": dev_right_metric.sentence_uf1_d,
                # "Ex_right_depth": dev_right_metric.sentence_ex_d,
            }

            for k, v in metric_dict.items():
                for i, val in v.items():
                    self.writer.add_scalar(f"{tag}/{k}", val, i)

            # distribution of estimated span depth
            self.estimated_depth = dict(sorted(self.estimated_depth.items()))
            for k, v in self.estimated_depth.items():
                self.writer.add_scalar(
                    'valid/estimated_depth', v/dev_f1_metric.n, k
                )

            t = datetime.now() - start

            # save the model if it is the best so far
            if dev_ll > best_metric:
                best_metric = dev_ll 
                best_e = epoch

                reproducibility.save(
                    args.save_dir + "/last.pt",
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    epoch=epoch,
                )
                log.info(f"{t}s elapsed (saved)\n")
            else:
                log.info(f"{t}s elapsed\n")

            # save the last model
            reproducibility.save(
                args.save_dir + "/last.pt",
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                epoch=epoch,
            )

            total_time += t
            if train_arg.patience > 0 and epoch - best_e >= train_arg.patience:
                if hasattr(self.train_arg, 'change') and self.train_arg.change:
                    self.train_arg.change = False
                    self.partition = not self.partition
                    best_metric = LikelihoodMetric()
                    best_metric.total_likelihood = -float('inf')
                    best_metric.total = 1
                else:
                    break
        
        self.writer.flush()
        self.writer.close()
