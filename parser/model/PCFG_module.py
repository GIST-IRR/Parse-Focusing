from argparse import ArgumentError
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import math

class PCFG_module(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def batch_dot(self, x, y):
        return (x*y).sum(-1, keepdims=True)

    num_trees_cache = {}
    def num_trees(self, len):
        if isinstance(len, torch.Tensor):
            len = len.item()
        if len == 1 or len == 2:
            return 1
        else:
            if len in self.num_trees_cache:
                num = self.num_trees_cache[len]
            else:
                num = 0
                for i in range(1, len):
                    num += self.num_trees(i) * self.num_trees(len-i)
                self.num_trees_cache[len] = num
            return num

    def max_entropy(self, num):
        return math.log(num)

    def cos_sim(self, x, log=False):
        b, cat = x.shape[:2]
        cs = x.new_zeros(b)
        for i in range(cat):
            if i == cat-1:
                continue
            u = x[:, i:i+1].expand(-1, cat-i-1, -1)
            o = x[:, i+1:cat]
            if not log:
                u = u.exp()
                o = o.exp()
            cosine_score = F.cosine_similarity(u, o, dim=2)
            cs += cosine_score.abs().sum(-1)
        cs = cs / (cat*(cat-1)/2)
        return cs

    def js_div(self, x, y, log_target=False):
        pass

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

    def update_depth(self, depth):
        self.depth = depth

    def clear_rules_grad(self):
        for k, v in self.rules.items():
            if k == 'kl':
                continue
            v.grad = None

    def get_grad(self):
        grad = []
        for p in self.parameters():
            grad.append(p.grad.reshape(-1))
        return torch.cat(grad)

    def set_grad(self, grad):
        total_num = 0
        for p in self.parameters():
            shape = p.grad.shape
            num = p.grad.numel()
            p.grad = p.grad + grad[total_num:total_num+num].reshape(*shape)
            total_num += num

    def get_rules_grad(self, flatten=False):
        b = 0
        grad = []
        for i, (k, v) in enumerate(self.rules.items()):
            if k == 'kl':
                continue
            if i == 0:
                b = v.shape[0]
            grad.append(v.grad)
        if flatten:
            grad = [g.reshape(b, -1) for g in grad]
        return grad

    def get_X_Y_Z(self, rule):
        NTs = slice(0, self.NT)
        return rule[:, :, NTs, NTs]

    def get_X_Y_z(self, rule):
        NTs = slice(0, self.NT)
        Ts = slice(self.NT, self.NT+self.T)
        return rule[:, :, NTs, Ts]

    def get_X_y_Z(self, rule):
        NTs = slice(0, self.NT)
        Ts = slice(self.NT, self.NT+self.T)
        return rule[:, :, Ts, NTs]

    def get_X_y_z(self, rule):
        Ts = slice(self.NT, self.NT+self.T)
        return rule[:, :, Ts, Ts]

    def get_rules_grad_category(self):
        b = 0
        grad = {}
        for i, (k, v) in enumerate(self.rules.items()):
            if k == 'kl':
                continue
            if i == 0:
                b = v.shape[0]
            g = v.grad
            if k == 'rule':
                g = g.reshape(b, g.shape[1], -1)
            grad[k] = g
        return grad

    def backward_rules(self, grad):
        total_num = 0
        for k, v in self.rules.items():
            if k == 'kl':
                continue
            shape = v.shape
            num = v[0].numel()
            v.backward(
                grad[:, total_num:total_num+num].reshape(*shape),
                retain_graph=True
            )
            total_num += num

    def backward_rules_category(self, grad):
        for k, v in grad.items():
            if k == 'rule':
                v = v.reshape(*self.rules[k].shape)
            self.rules[k].backward(
                v,
                retain_graph=True
            )
        
    def term_from_unary(self, input, term):
        x = input['word']
        n = x.shape[1]
        b = term.shape[0]
        term = term.unsqueeze(1).expand(b, n, self.T, self.V)
        indices = x[..., None, None].expand(b, n, self.T, 1)
        return torch.gather(term, 3, indices).squeeze(3)

    def soft_backward(self, loss, z_l, optimizer, dambda=1.0, target='rule', mode='projection'):
        def batch_dot(x, y):
            return (x*y).sum(-1, keepdims=True)
        def projection(x, y):
            scale = (batch_dot(x, y)/batch_dot(y, y))
            return scale * y, scale
        # Get dL_w
        loss.backward(retain_graph=True)
        if target == 'rule':
            g_loss = self.get_rules_grad() # main vector
            # g_loss = self.get_rules_grad_category()
            # self.save_rule_heatmap(g_loss[-1][0], dirname='figure', filename='loss_gradient.png', abs=False, symbol=False)
            self.clear_rules_grad()
        elif target == 'parameter':
            g_loss = self.get_grad()
            g_loss_norm = batch_dot(g_loss, g_loss).sqrt()
        optimizer.zero_grad()
        # Get dZ_l
        z_l.backward(retain_graph=True)
        if target == 'rule':
            g_z_l = self.get_rules_grad()
            # g_z_l = self.get_rules_grad_category()
            # self.save_rule_heatmap(g_z_l[-1][0], dirname='figure', filename='z_gradient.png', abs=False, symbol=False)
            self.clear_rules_grad()
        elif target == 'parameter':
            g_z_l = self.get_grad()
            g_z_l_norm = batch_dot(g_z_l, g_z_l).sqrt()
        optimizer.zero_grad()

        # if target == 'parameter':
        #     g_rule = self.get_rules_grad()
        #     self.save_rule_heatmap(g_rule[-1][0], dirname='figure', filename='rule_gradient.png', abs=False, symbol=False)

        # tmp
        # TODO: remove unused computing
        # loss.backward(retain_graph=True)
        # grad_output = torch.tensor(dambda)
        # z_l.backward(grad_output, retain_graph=True)
        # tmp_g_z_l = self.get_grad()
        # optimizer.zero_grad()

        if mode == 'both':
            if target == 'rule':
                g_r = [g_l + dambda * g_z for g_l, g_z in zip(g_loss, g_z_l)]
                # self.save_rule_heatmap(g_r[-1][0], dirname='figure', filename='rule_gradient.png', abs=False, symbol=False)
            elif target == 'parameter':
                g_r = g_loss + dambda * g_z_l
        elif mode == 'projection':
            g_proj, proj_scale = projection(g_z_l, g_loss)
            g_orth = g_z_l - g_proj
            g_proj_norm = batch_dot(g_proj, g_proj).sqrt()
            g_orth_norm = batch_dot(g_orth, g_orth).sqrt()
            g_r = g_loss + g_proj + dambda * g_orth
            # g_r = g_loss + dambda * g_z_l
            # g_r = {}
            # for k, v in dambda.items():
            #     if g_z_l[k].dim() == 3:
            #         v = v[None, :, None]
            #     g_r[k] = g_loss[k] + v * g_z_l[k]
        elif mode == 'orthogonal':
        # oproj_{dL_w}{dZ_l} = dZ_l - proj_{dL_w}{dZ_l}
            g_oproj = g_z_l - projection(g_z_l, g_loss)
        # dL_BCLs = dL_w + oproj_{dL_w}{dZ_l}
            g_r = g_loss + g_oproj

        # Re-calculate soft BCL
        if target == 'rule':
            # self.backward_rules_category(g_r)
            # b = g_loss['root'].shape[0]
            # g_loss = torch.cat([g.reshape(b, -1) for g in g_loss.values()], dim=-1)
            # g_z_l = torch.cat([g.reshape(b, -1) for g in g_z_l.values()], dim=-1)
            # g_r = torch.cat([g.reshape(b, -1) for g in g_r.values()], dim=-1)
            self.backward_rules(g_r)
        elif target == 'parameter':
            # grad_norm = g_orth_norm.mean()
            # grad_norm.backward()
            self.set_grad(g_r)

        return {
            'g_loss': g_loss,
            'g_z_l': g_z_l,
            'g_r': g_r,
            # 'proj_scale': proj_scale,
            'g_loss_norm': g_loss_norm,
            'g_z_l_norm': g_z_l_norm
            # 'g_proj_norm': g_proj_norm,
            # 'g_orth_norm': g_orth_norm
        }