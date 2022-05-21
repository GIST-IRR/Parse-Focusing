import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG

class CompoundPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(CompoundPCFG, self).__init__()
        self.pcfg = PCFG()
        self.part = PartitionFunction()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.w_dim = args.w_dim
        self.h_dim = args.h_dim

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        input_dim = self.s_dim + self.z_dim

        self.term_mlp = nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.root_mlp = nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        self.enc_emb = nn.Embedding(self.V, self.w_dim)

        self.enc_rnn = nn.LSTM(self.w_dim, self.h_dim, bidirectional=True, num_layers=1, batch_first=True, device=self.device)

        self.enc_out = nn.Linear(self.h_dim * 2, self.z_dim * 2)

        self.NT_T = self.NT + self .T
        self.rule_mlp = nn.Linear(input_dim, (self.NT_T) ** 2)
        # Partition function
        self.mode = args.mode if hasattr(args, 'mode') else None
        # Fix embedding vectors of symbols
        # if hasattr(args, 'fix_root'):
        #     self.root_emb.requires_grad = args.fix_root
        # if hasattr(args, 'fix_nonterm'):
        #     self.nonterm_emb.requires_grad = args.fix_nonterm
        # if hasattr(args, 'fix_term'):
        #     self.term_emb.requires_grad = args.fix_term

        # if hasattr(args, 'fix_root_mlp') and args.fix_root_mlp:
        #     for p in self.root_mlp.parameters():
        #         p.requires_grad = False
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_switch(self, switch: bool):
        self.switch = switch

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
        b = self.rules['root'].shape[0]
        return torch.cat([
            self.rules['root'].grad.clone().reshape(b, -1),
            self.rules['rule'].grad.clone().reshape(b, -1),
            self.rules['unary'].grad.clone().reshape(b, -1)
            # self.rules['unary_full'].grad.clone().reshape(b, -1)
        ], dim=-1)

    def backward_rules(self, grad):
        rs = ['root', 'rule', 'unary']
        total_num = 0
        for r in rs:
            shape = self.rules[r].shape
            num = self.rules[r][0].numel()
            self.rules[r].backward(grad[:, total_num:total_num+num].reshape(*shape), retain_graph=True)
            total_num += num

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]
        seq_len = input['seq_len']

        def enc(x):
            x_embbed = self.enc_emb(x)
            x_packed = pack_padded_sequence(
                x_embbed, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.enc_rnn(x_packed)
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
            out = self.enc_out(h)
            mean = out[:, : self.z_dim]
            lvar = out[:, self.z_dim :]
            return mean, lvar

        def kl(mean, logvar):
            result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
            return result

        mean, lvar = enc(x)
        z = mean

        if not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0,1)
            z = (0.5 * lvar).exp() * z + mean


        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            root_emb = torch.cat([root_emb, z], -1)
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots

        def terms():
            #TODO: Check the replication between embs before mlp
            # term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
            #     b, n, self.T, self.s_dim
            # )
            # z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
            # z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
            # term_emb = torch.cat([term_emb, z_expand], -1)
            # term_prob = self.term_mlp(term_emb).log_softmax(-1)
            # indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            # term_prob = torch.gather(term_prob, 3, indices).squeeze(3)

            term_emb = self.term_emb.unsqueeze(0).expand(
                b, self.T, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(b, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(
                b, self.NT, self.z_dim
            )
            nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            rule_prob = self.rule_mlp(nonterm_emb).log_softmax(-1)
            rule_prob = rule_prob.reshape(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        root, unary, rule = roots(), terms(), rules()

        # for gradient conflict by using gradients of rules
        if self.training:
            root.retain_grad()
            unary.retain_grad()
            rule.retain_grad()

        return {'unary': unary,
                'root': root,
                'rule': rule,
                'kl': kl(mean, lvar).sum(1)}

    def term_from_unary(self, input, term):
        x = input['word']
        n = x.shape[1]
        b = term.shape[0]
        term = term.unsqueeze(1).expand(b, n, self.T, self.V)
        indices = x[..., None, None].expand(b, n, self.T, 1)
        return torch.gather(term, 3, indices).squeeze(3)

    def loss(self, input, partition=False, max_depth=0, soft=False):
        self.rules = self.forward(input)
        terms = self.term_from_unary(input, self.rules['unary'])

        result = self.pcfg(self.rules, terms, lens=input['seq_len'], depth=False)
        # Partition function
        if partition:
            # depth-conditioned inside algorithm
            # partition function approximation
            # if max_depth == 0:
            #     lens = input['seq_len']
            # else:
            #     lens = max_depth
            self.pf = self.part(self.rules, lens=input['seq_len'], mode=self.mode)
            # Renormalization
            if soft:
                return (-result['partition'] + self.rules['kl']).mean(), self.pf.mean()
            result['partition'] = result['partition'] - self.pf
        # depth-conditioned inside algorithm
        loss =  (-result['partition'] + self.rules['kl']).mean()
        return loss

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
        g_oproj = g_z_l - projection(g_z_l, g_loss)
        # dL_BCLs = dL_w + oproj_{dL_w}{dZ_l}
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
        terms = self.term_from_unary(input, rules['unary'])

        if decode_type == 'viterbi':
            result = self.pcfg.decode(rules, terms, lens=input['seq_len'], viterbi=True, mbr=False, depth=depth_mode)
        elif decode_type == 'mbr':
            result = self.pcfg.decode(rules, terms, lens=input['seq_len'], viterbi=False, mbr=True, depth=depth_mode)
        else:
            raise NotImplementedError

        if depth > 0:
            result['depth'] = self.pcfg._partition_function(rules, depth, mode='length', depth_output='full')
            result['depth'] = result['depth'].exp()
            
        result['partition'] -= rules['kl']
        return result
