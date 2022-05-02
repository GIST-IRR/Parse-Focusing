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
        best_model_path = self.args.load_from_dir + "/best.pt"
        self.model.load_state_dict(torch.load(str(best_model_path), map_location=self.device))
        print('successfully load')

        eval_depth = self.args.test.eval_depth if hasattr(self.args.test, 'eval_depth') else False
        left_binarization = self.args.test.left_binarization if hasattr(self.args.test, 'left_binarization') else False
        right_binarization = self.args.test.right_binarization if hasattr(self.args.test, 'right_binarization') else False
        self.writer = SummaryWriter(self.args.load_from_dir)

        test_loader = dataset.test_dataloader
        test_loader_autodevice = DataPrefetcher(test_loader, device=self.device)

        metric_f1, metric_uas, likelihood, metric_left, metric_right = self.evaluate(
            test_loader_autodevice, eval_dep=eval_dep, decode_type=decode_type,
            eval_depth=eval_depth, left_binarization=left_binarization, right_binarization=right_binarization
        )
        try:
            print(metric_uas)
        except:
            print('No UAS')
        print(metric_f1)
        print(likelihood)

        # Log - Tensorboard
        for i, pf in enumerate(self.pf_sum):
            self.writer.add_scalar('test/partition_number', pf/metric_f1.n, i)
        for k, v in metric_f1.sentence_uf1_d.items():
            self.writer.add_scalar('test/f1_depth', v, k)
        # for k, v in metric_f1.sentence_uf1_l.items():
        #     self.writer.add_scalar('test/f1_length', v, k)
        for k, v in metric_f1.sentence_ex_d.items():
            self.writer.add_scalar('test/Ex_depth', v, k)

        # F1 score for each depth
        for k, v in metric_left.sentence_uf1_d.items():
            self.writer.add_scalar('test/f1_left_depth', v, k)
        # F1 score for each length
        # for k, v in metric_left.sentence_uf1_l.items():
        #     self.writer.add_scalar('valid/f1_left_length', v, k)
        for k, v in metric_left.sentence_ex_d.items():
            self.writer.add_scalar('test/Ex_left_depth', v, k)

        # F1 score for each depth
        for k, v in metric_right.sentence_uf1_d.items():
            self.writer.add_scalar('test/f1_right_depth', v, k)
        # F1 score for each length
        # for k, v in metric_right.sentence_uf1_l.items():
        #     self.writer.add_scalar('valid/f1_right_length', v, k)
        for k, v in metric_right.sentence_ex_d.items():
            self.writer.add_scalar('test/Ex_right_depth', v, k)

        self.estimated_depth = dict(sorted(self.estimated_depth.items()))
        for k, v in self.estimated_depth.items():
            self.writer.add_scalar('test/estimated_depth', v/metric_f1.n, k)

        self.estimated_depth_by_length = dict(sorted(self.estimated_depth_by_length.items()))
        for k in self.estimated_depth_by_length.keys():
            self.estimated_depth_by_length[k] = dict(sorted(self.estimated_depth_by_length[k].items()))
        for k, v in self.estimated_depth_by_length.items():
            total = sum(v.values())
            for d, n in v.items():
                self.writer.add_scalar(f'test/predicted_depth_by_length_{k}', n/total, d)

        self.writer.flush()
        self.writer.close()
        
        # Log - CSV
        import csv
        label_recall = self.args.test.label_recall if hasattr(self.args.test, 'label_recall') else False
        if label_recall:
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
