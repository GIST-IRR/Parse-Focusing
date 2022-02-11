# -*- coding: utf-8 -*-


from parser.cmds.cmd import CMD

from datetime import datetime, timedelta
from parser.cmds.cmd import CMD
from parser.helper.metric import Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch
import numpy as np
from parser.helper.util import *
from parser.helper.data_module import DataModule
import click

from torch.utils.tensorboard import SummaryWriter

class Evaluate(CMD):

    def __call__(self, args, eval_dep=False, decode_type='mbr'):
        super(Evaluate, self).__call__(args)
        self.device = args.device
        self.args = args
        dataset = DataModule(args)
        self.model = get_model(args.model, dataset)
        if hasattr(args.test, 'depth'):
            self.model.depth = args.test.depth
        elif hasattr(args.train, 'min_depth'):
            self.model.depth = args.train.min_depth
        best_model_path = self.args.load_from_dir + "/best.pt"
        self.model.load_state_dict(torch.load(str(best_model_path)))
        print('successfully load')

        self.writer = SummaryWriter(self.args.load_from_dir)

        test_loader = dataset.test_dataloader
        test_loader_autodevice = DataPrefetcher(test_loader, device=self.device)
        if not eval_dep:
            metric_f1, likelihood = self.evaluate(test_loader_autodevice, eval_dep=eval_dep, decode_type=decode_type)
        else:
            metric_f1, metric_uas, likelihood = self.evaluate(test_loader_autodevice, eval_dep=eval_dep, decode_type=decode_type)
            print(metric_uas)
        print(metric_f1)
        print(likelihood)

        # Log - Tensorboard
        for i, pf in enumerate(self.pf_sum):
            self.writer.add_scalar('test/partition_number', pf/metric_f1.n, i)
        for k, v in metric_f1.sentence_uf1_d.items():
            self.writer.add_scalar('test/f1_depth', v, k)

        self.span_depth = dict(sorted(self.span_depth.items()))
        for k, v in self.span_depth.items():
            self.writer.add_scalar('test/span_depth', v/metric_f1.n, k)

        self.writer.flush()
        self.writer.close()
        
        # Log - CSV
        import csv
        nt_log = os.path.join(self.args.load_from_dir, 'label_recall.csv')
        nt_label = ['SBAR', 'NP', 'VP', 'PP', 'ADJP', 'ADVP']
        with open(nt_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(nt_label)
            writer.writerow([metric_f1.label_recall[l] for l in nt_label])

        log_file = os.path.join(os.path.dirname(args.load_from_dir), 'test.csv')
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                args.load_from_dir, metric_f1.sentence_uf1, metric_f1.corpus_uf1,
                likelihood.avg_likelihood.item(), likelihood.perplexity.item()
            ])
