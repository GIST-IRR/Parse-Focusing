import torch
import torch.nn as nn
from parser.modules.res import ResLayer

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG

class NeuralPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()
        self.device = dataset.device
        self.args = args

        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        self.NT_T = self.NT + self.T
        self.rule_mlp = nn.Linear(self.s_dim, (self.NT_T) ** 2)
        # Partition function
        self.mode = args.mode if hasattr(args, 'mode') else None

        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_depth(self, depth):
        self.depth = depth

    def get_grad(self):
        # grad = []
        # for p in self.parameters():
        #     grad.append(p.grad.clone())
        # return grad
        grad = []
        for p in self.parameters():
            grad.append(p.grad.clone().reshape(-1))
        return torch.cat(grad)

    def set_grad(self, grad):
        total_num = 0
        for p in self.parameters():
            num = p.grad.numel()
            shape = p.grad.shape
            p.grad = p.grad + grad[total_num:total_num+num].reshape(*shape)
            total_num += num

    def get_rules_grad(self):
        b = self.rules['root'].shape[0]
        return torch.cat([
            self.rules['root'].grad.clone().reshape(b, -1),
            self.rules['rule'].grad.clone().reshape(b, -1),
            self.rules['unary'].grad.clone().reshape(b, -1)
        ], dim=-1)

    def backward_rules(self, grad):
        rule_shape = self.rules['rule'].shape
        unary_shae = self.rules['unary'].shape
        self.rules['root'].backward(grad[:, :self.NT], retain_graph=True)
        self.rules['rule'].backward(grad[:, self.NT:self.NT*(1+self.NT_T**2)].reshape(*rule_shape), retain_graph=True)
        self.rules['unary'].backward(grad[:, self.NT*(1+self.NT_T**2):].reshape(*unary_shae), retain_graph=True)

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots.expand(b, self.NT)

        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            )
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            rule_prob = self.rule_mlp(self.nonterm_emb).log_softmax(-1)
            rule_prob = rule_prob.reshape(self.NT, self.NT_T, self.NT_T)
            return rule_prob.unsqueeze(0).expand(b, *rule_prob.shape).contiguous()

        root, unary, rule = roots(), terms(), rules()

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            unary.retain_grad()
            rule.retain_grad()

        return {'unary': unary,
                'root': root,
                'rule': rule,
                'kl': torch.tensor(0, device=self.device)}


    def loss(self, input, partition=False, soft=False):
        self.rules = self.forward(input)
        result = self.pcfg(self.rules, lens=input['seq_len'])
        if partition:
            self.pf = self.part(self.rules, lens=input['seq_len'], mode=self.mode)
            if soft:
                return -result['partition'].mean(), self.pf.mean()
            result['partition'] = result['partition'] - self.pf
        return -result['partition'].mean()

    def rule_backward(self, loss, z_l, optimizer):
        def batch_dot(x, y):
            return (x*y).sum(-1, keepdims=True)
        def projection(x, y):
            return (batch_dot(x, y)/batch_dot(y, y))*y
        # Get dL_w
        loss.backward(retain_graph=True)
        g_loss = self.get_rules_grad() # main vector
        optimizer.zero_grad()
        # Get dZ_l
        z_l.backward(retain_graph=True)
        g_z_l = self.get_rules_grad()
        optimizer.zero_grad()
        # oproj_{dL_w}{dZ_l} = dZ_l - proj_{dL_w}{dZ_l}
        g_oproj = g_z_l - projection(g_z_l, g_loss)
        # dL_BCLs = dL_w + oproj_{dL_w}{dZ_l}
        g_r = g_loss + g_oproj
        # Re-calculate soft BCL
        self.backward_rules(g_r)

    def param_backward(self, loss, z_l, optimizer):
        def batch_dot(x, y):
            return (x*y).sum(-1, keepdims=True)
        def projection(x, y):
            eps = 1e-9
            return (batch_dot(x, y)/(batch_dot(y, y)+eps))*y
        # Get dL_w
        loss.backward(retain_graph=True)
        g_loss = self.get_grad() # main vector
        optimizer.zero_grad()
        # Get dZ_l
        z_l.backward(retain_graph=True)
        g_z_l = self.get_grad()
        optimizer.zero_grad()

        # oproj_{dL_w}{dZ_l} = dZ_l - proj_{dL_w}{dZ_l}
        # dL_BCLs = dL_w + oproj_{dL_w}{dZ_l}
        # g_r = [g_l + g_z - projection(g_z, g_l) \
        #     for g_l, g_z in zip(g_loss, g_z_l)]
        g_oproj = g_z_l - projection(g_z_l, g_loss)
        g_r = g_loss + g_oproj
        # Re-calculate soft BCL
        self.set_grad(g_r)

    def soft_backward(self, loss, z_l, optimizer, mode='rule'):
        if mode == 'rule':
            self.rule_backward(loss, z_l, optimizer)
        elif mode == 'parameter':
            self.param_backward(loss, z_l, optimizer)


    def evaluate(self, input, decode_type, depth=0, depth_mode=False, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        if depth > 0:
            result['depth'] = self.pcfg._partition_function(rules, depth, mode='length', depth_output='full')
            result['depth'] = result['depth'].exp()

        return result