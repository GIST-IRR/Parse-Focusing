from argparse import ArgumentError
import torch
import torch.nn as nn
import torch.distributions as dist

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

    def get_max_entropy(self):
        raise NotImplementedError('Add to each PCFGs')

    def entropy_rules(self):
        raise NotImplementedError('Add to each PCFGs')

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

    def get_rules_grad(self):
        b = 0
        grad = []
        for i, (k, v) in enumerate(self.rules.items()):
            if k == 'kl':
                continue
            if i == 0:
                b = v.shape[0]
            grad.append(v.grad.reshape(b, -1))
        return grad

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
            self.clear_rules_grad()
        elif target == 'parameter':
            g_z_l = self.get_grad()
            g_z_l_norm = batch_dot(g_z_l, g_z_l).sqrt()
        optimizer.zero_grad()

        if mode == 'both':
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
            # z_norm = batch_dot(g_z_l, g_z_l).sqrt()
            # total_loss = (loss + z_l + proj_scale + z_norm).mean()
            # total_loss.backward()
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