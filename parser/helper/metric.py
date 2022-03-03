# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, OrderedDict
import torch


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return -1e9

class UF1(Metric):
    def __init__(self, eps=1e-8, device=torch.device("cuda")):
        super(UF1, self).__init__()
        self.f1 = 0.0
        self.evalb = 0.0
        self.n = 0.0
        self.eps = eps
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.ex = 0.0
        self.depth_n = {}
        self.depth_f1 = {}
        self.depth_ex = {}
        self.length_n = {}
        self.length_f1 = {}
        self.length_ex = {}
        self.nt_tp = {}
        self.nt_fn = {}
        self.z = 0.0
        self.device = device


    def __call__(self, preds, golds, depth=None, lens=False, nonterminal=False):
        if depth:
            zipped = zip(preds, golds, depth)
        else:
            zipped = zip(preds, golds)
        for e in zipped:
            if depth:
                pred, gold, d = e
            else: 
                pred, gold = e
            # in the case of sentence length=1
            if len(pred) == 0:
                continue
            length = max(gold,key=lambda x:x[1])[1]
            #removing the trival span
            gold = list(filter(lambda x: x[0]+1 != x[1], gold))
            pred = list(filter(lambda x: x[0]+1 != x[1], pred))
            #remove the entire sentence span.
            gold = list(filter(lambda x: not (x[0]==0 and x[1]==length), gold))
            pred = list(filter(lambda x: not (x[0]==0 and x[1]==length), pred))
            #remove label.
            if nonterminal:
                label = [g[2] for g in gold]
            gold = [g[:2] for g in gold]
            pred = [p[:2] for p in pred]
            gold = list(map(tuple, gold))
            #corpus f1
            for span in pred:
                if span in gold:
                    self.tp += 1
                    l = label[gold.index(span)]
                    if l in self.nt_tp:
                        self.nt_tp[l] += 1
                    else:
                        self.nt_tp[l] = 1
                else:
                    self.fp += 1
            for span in gold:
                if span not in pred:
                    self.fn += 1
                    l = label[gold.index(span)]
                    if l in self.nt_fn:
                        self.nt_fn[l] += 1
                    else:
                        self.nt_fn[l] = 1

            #sentence f1
            #remove duplicated span.
            gold = set(gold)
            pred = set(pred)
            overlap = pred.intersection(gold)
            prec = float(len(overlap)) / (len(pred) + self.eps)
            reca = float(len(overlap)) / (len(gold) + self.eps)
            if len(gold) == 0:
                reca = 1.
                if len(pred) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            ex = 1 if (1 - f1) < 1e-8 else 0
            self.f1 += f1
            self.ex += ex
            self.n += 1
            if depth:
                if d in self.depth_f1:
                    self.depth_f1[d] += f1
                    self.depth_ex[d] += ex
                    self.depth_n[d] += 1
                else:
                    self.depth_f1[d] = f1
                    self.depth_ex[d] = ex
                    self.depth_n[d] = 1
            if lens:
                if length in self.length_f1:
                    self.length_f1[length] += f1
                    self.length_ex[length] += ex
                    self.length_n[length] += 1
                else:
                    self.length_f1[length] = f1
                    self.length_ex[length] = ex
                    self.length_n[length] = 1

    @property
    def sentence_uf1(self):
        return self.f1 / self.n

    @property
    def corpus_uf1(self):
        if self.tp == 0 and self.fp == 0:
            return 0

        prec = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
        return corpus_f1

    @property
    def sentence_uf1_d(self):
        self.depth_f1 = dict(sorted(self.depth_f1.items()))
        result = {}
        for d, f1 in self.depth_f1.items():
            result[d] = f1 / self.depth_n[d]
        return result
    
    @property
    def sentence_uf1_l(self):
        self.length_f1 = dict(sorted(self.length_f1.items()))
        result = {}
        for l, f1 in self.length_f1.items():
            result[l] = f1 / self.length_n[l]
        return result

    @property
    def label_recall(self):
        result = {}
        for l, tp in self.nt_tp.items():
            fn = self.nt_fn[l] if l in self.nt_fn else 0
            result[l] = tp / (tp + fn)
        return result

    @property
    def partition_number(self):
        return self.z / self.n

    @property
    def score(self):
        return self.sentence_uf1

    def __repr__(self):
        s = f"Sentence F1: {self.sentence_uf1:6.2%} Corpus F1: {self.corpus_uf1:6.2%} "
        return s

class UAS(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.direct_correct = 0.0
        self.undirect_correct = 0.0
        self.total_sentence = 0.0
        self.correct_root = 0.0

    @property
    def score(self):
        return   self.direct_correct / self.total

    def __call__(self, predicted_arcs, gold_arcs):

        for pred, gold in zip(predicted_arcs, gold_arcs):
            assert len(pred) == len(gold)

            if len(pred) > 0:
                self.total_sentence+=1.

            for (head, child) in pred:
                if gold[int(child)] == int(head) + 1:
                    self.direct_correct += 1.
                    self.undirect_correct += 1.
                    if int(head) + 1 == 0:
                        self.correct_root += 1.

                elif gold[int(head)] == int(child) + 1:
                    self.undirect_correct += 1.



                self.total += 1.


    def __repr__(self):
        return "UDAS: {}, UUAS:{}, root:{} ".format(self.score, self.undirect_correct/self.total, self.correct_root/self.total_sentence)


class LossMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.total_likelihood = 0.0
        self.total_kl = 0.0
        self.calling_time = 0


    def __call__(self, likelihood):
        self.calling_time += 1
        self.total += likelihood.shape[0]
        self.total_likelihood += likelihood.detach_().sum()

    @property
    def avg_loss(self):
        return self.total_likelihood / self.total


    def __repr__(self):
        return "avg likelihood: {} kl: {}, total likelihood:{}, n:{}".format(self.avg_likelihood,self.avg_kl,self.total_likelihood, self.total)

    @property
    def score(self):
        return (self.avg_likelihood + self.avg_kl).item()


class LikelihoodMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.total = 0.0
        self.total_likelihood = 0.0
        self.total_word = 0

    @property
    def score(self):
        return self.avg_likelihood


    def __call__(self, likelihood, lens):

        self.total += likelihood.shape[0]
        self.total_likelihood += likelihood.detach_().sum()
        # Follow Yoon Kim
        self.total_word += (lens.sum() + lens.shape[0])

    @property
    def avg_likelihood(self):
        return self.total_likelihood / self.total


    @property
    def perplexity(self):
        return (-self.total_likelihood / self.total_word).exp()

    def __repr__(self):
        return "avg likelihood: {}, perp. :{}".format(self.avg_likelihood, self.perplexity)



