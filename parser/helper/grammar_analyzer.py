import math
import torch.nn.functional as F
import torch.distributions as dist
from argparse import ArgumentError


class GrammarAnalyzer:
    
    def __init__(self, grammar) -> None:
        self.grammar = grammar

    def max_entropy(self, num):
        return math.log(num)

    def cos_sim(self, x, log=False):
        b, cat = x.shape[:2]
        cs = x.new_zeros(cat, cat).fill_diagonal_(1).expand(b, -1, -1).clone()
        for i in range(cat):
            if i == cat-1:
                continue
            u = x[:, i:i+1].expand(-1, cat-i-1, -1)
            o = x[:, i+1:cat]
            if not log:
                u = u.exp()
                o = o.exp()
            cosine_score = F.cosine_similarity(u, o, dim=2)
            cs[:, i, i+1:cat] = cosine_score
            cs[:, i+1:cat, i] = cosine_score
        return cs

    def kl_div(self, x):
        b, cat = x.shape[:2]
        kl = x.new_zeros(b, cat, cat)
        x = x.reshape(b, cat, -1)
        for i in range(cat):
            t = x[:, i:i+1].expand(-1, cat, -1)
            kl_score = F.kl_div(t, x, log_target=True, reduction='none')
            kl_score = kl_score.sum(-1)
            kl[:, i] = kl_score
        # reverse ratio of kl score
        # mask = nkl.new_ones(nkl.shape[1:]).fill_diagonal_(0)
        # weight = 1 - (nkl / nkl.sum((1, 2), keepdims=True))
        # weight = (mask * weight).detach()
        # nkl = (weight * nkl).mean((1, 2))
        # nkl = nkl.mean()
        return kl

    def cos_sim_mean(self, x):
        cat = x.shape[1]
        x = x.tril(diagonal=-1)
        x = x.flatten(start_dim=1)
        x = x.abs()
        x = x.sum(-1)
        x = x / (cat*(cat-1)/2)
        return x

    def cos_sim_max(self, x):
        x = x.tril(diagonal=-1)
        x = x.flatten(start_dim=1)
        x = x.abs()
        return x.max(-1)[0]

    def js_div(self, x, y, log_target=False):
        raise NotImplementedError

    def _entropy(self, rule, batch=False, reduce='none', probs=False):
        if rule.dim() == 2:
            rule = rule.unsqueeze(1)
        elif rule.dim() == 3:
            pass
        elif rule.dim() == 4:
            rule = rule.reshape(*rule.shape[:2], -1)
        else:
            raise ArgumentError(
                f'Wrong size of rule tensor. The allowed size is (2, 3, 4), but given tensor is {rule.dim()}'
            )

        b, n_parent, n_children = rule.shape
        if batch:
            ent = rule.new_zeros((b, n_parent))
            for i in range(b):
                for j in range(n_parent):
                    ent[i, j] = dist.categorical.Categorical(logits=rule[i, j]).entropy()
        else:
            rule = rule[0]
            ent = rule.new_zeros((n_parent, ))
            for i in range(n_parent):
                ent[i] = dist.categorical.Categorical(logits=rule[i]).entropy()

        if reduce == 'mean':
            ent = ent.mean(-1)
        elif reduce == 'sum':
            ent = ent.sum(-1)

        if probs:
            emax = self.max_entropy(n_children)
            ent = (emax - ent) / emax
        
        return ent