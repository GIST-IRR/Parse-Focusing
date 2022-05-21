import torch
import torch.nn as nn
from parser.modules.res import ResLayer

from parser.pcfgs.td_partition_function import TDPartitionFunction
from ..pcfgs.tdpcfg import TDPCFG

class TNPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(TNPCFG, self).__init__()
        self.pcfg = TDPCFG()
        self.part = TDPartitionFunction()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)
        self.s_dim = args.s_dim
        self.r = args.r_dim
        self.word_emb_size = args.word_emb_size

        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        #terms
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT+self.T, self.s_dim))
        rule_dim = self.s_dim
        self.parent_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU(),nn.Linear(rule_dim,self.r))
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim), nn.ReLU(),nn.Linear(rule_dim,self.r))
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU(),nn.Linear(rule_dim,self.r))

        # Partition function
        self.mode = args.mode if hasattr(args, 'mode') else None

    def update_depth(self, depth):
        self.depth = depth

    def get_grad(self):
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
            self.rules['head'].grad.clone().reshape(b, -1),
            self.rules['left'].grad.clone().reshape(b, -1),
            self.rules['right'].grad.clone().reshape(b, -1),
            self.rules['unary'].grad.clone().reshape(b, -1)
        ], dim=-1)

    def backward_rules(self, grad):
        total_num = 0
        for r in ['root', 'head', 'left', 'right', 'unary']:
            shape = self.rules[r].shape
            size = self.rules[r][0].numel()
            self.rules[r].backward(grad[:, total_num:total_num+size].reshape(*shape), retain_graph=True)
            total_num += size

    def forward(self, input, **kwargs):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            roots = self.root_mlp(self.root_emb).log_softmax(-1)
            return roots.expand(b, roots.shape[-1]).contiguous()

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            # term_prob = term_prob.unsqueeze(0).unsqueeze(1).expand(
            #     b, n, self.T, self.V
            # )
            term_prob = term_prob.unsqueeze(0).expand(b, self.T, self.V)
            # indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            # term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[:self.NT]
            head = self.parent_mlp(nonterm_emb).log_softmax(-1)
            left = self.left_mlp(rule_state_emb).log_softmax(-2)
            right = self.right_mlp(rule_state_emb).log_softmax(-2)
            head = head.unsqueeze(0).expand(b,*head.shape)
            left = left.unsqueeze(0).expand(b,*left.shape)
            right = right.unsqueeze(0).expand(b,*right.shape)
            return (head, left, right)

        root, unary, (head, left, right) = roots(), terms(), rules()

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            unary.retain_grad()
            head.retain_grad()
            left.retain_grad()
            right.retain_grad()

        return {'unary': unary,
                'root': root,
                'head': head,
                'left': left,
                'right': right,
                'kl': 0}

    def term_from_unary(self, input, term):
        x = input['word']
        n = x.shape[1]
        b = term.shape[0]
        term = term.unsqueeze(1).expand(b, n, self.T, self.V)
        indices = x[..., None, None].expand(b, n, self.T, 1)
        return torch.gather(term, 3, indices).squeeze(3)

    def loss(self, input, partition=False, soft=False):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input, self.rules['unary'])

        result = self.pcfg(self.rules, terms, lens=input['seq_len'])
        # Partition function
        if partition:
            self.pf = self.part(self.rules, input['seq_len'], mode=self.mode)
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
            return (batch_dot(x, y)/(batch_dot(y, y)))*y
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


    def evaluate(self, input, decode_type, depth=0, **kwargs):
        rules = self.forward(input)
        terms = self.term_from_unary(input, rules['unary'])

        if decode_type == 'viterbi':
            assert NotImplementedError
        elif decode_type == 'mbr':
            result = self.pcfg(rules, terms, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        if depth > 0:
            result['depth'] = self.part(rules, depth, mode='length', depth_output='full')
            result['depth'] = result['depth'].exp()

        return result