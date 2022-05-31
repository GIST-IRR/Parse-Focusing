import torch
import torch.nn as nn

class PCFG_module(nn.Module):
    def __init__(self) -> None:
        super().__init__()

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
            grad.append(p.grad.clone().reshape(-1))
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
            grad.append(v.grad.clone().reshape(b, -1))
        return torch.cat(grad, dim=-1)

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
        
    def term_from_unary(self, input, term):
        x = input['word']
        n = x.shape[1]
        b = term.shape[0]
        term = term.unsqueeze(1).expand(b, n, self.T, self.V)
        indices = x[..., None, None].expand(b, n, self.T, 1)
        return torch.gather(term, 3, indices).squeeze(3)

    def soft_backward(self, loss, z_l, optimizer, target='rule', mode='projection'):
        dambda = 1.0
        def batch_dot(x, y):
            return (x*y).sum(-1, keepdims=True)
        def projection(x, y):
            scale = (batch_dot(x, y)/batch_dot(y, y))
            return scale * y, scale
        # Get dL_w
        loss.backward(retain_graph=True)
        if target == 'rule':
            g_loss = self.get_rules_grad() # main vector
            self.clear_rules_grad()
        elif target == 'parameter':
            g_loss = self.get_grad()
            optimizer.zero_grad()
        # Get dZ_l
        z_l.backward(retain_graph=True)
        if target == 'rule':
            g_z_l = self.get_rules_grad()
            self.clear_rules_grad()
        elif target == 'parameter':
            g_z_l = self.get_grad()
            optimizer.zero_grad()

        if mode == 'projection':
            g_proj, proj_scale = projection(g_z_l, g_loss)
            g_r = g_loss + dambda * g_proj
        elif mode == 'orthogonal':
        # oproj_{dL_w}{dZ_l} = dZ_l - proj_{dL_w}{dZ_l}
            g_oproj = g_z_l - projection(g_z_l, g_loss)
        # dL_BCLs = dL_w + oproj_{dL_w}{dZ_l}
            g_r = g_loss + g_oproj
        # Re-calculate soft BCL
        if target == 'rule':
            self.backward_rules(g_r)
        elif target == 'parameter':
            self.set_grad(g_r)

        return {
            'g_loss': g_loss,
            'g_z_l': g_z_l,
            'g_r': g_r,
            'proj_scale': proj_scale
        }