import torch
import torch.nn as nn
from parser.model.PCFG_module import PCFG_module
from parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from parser.pcfgs.partition_function import PartitionFunction
from ..pcfgs.pcfg import PCFG

class CompoundPCFG(PCFG_module):
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
        return (-result['partition'] + self.rules['kl']).mean()

    def evaluate(self, input, decode_type, depth=0, depth_mode=False, **kwargs):
        rules = self.forward(input, evaluating=True)
        terms = self.term_from_unary(input, rules['unary'])

        if decode_type == 'viterbi':
            result = self.pcfg(rules, terms, lens=input['seq_len'], viterbi=True, mbr=False, depth=depth_mode)
        elif decode_type == 'mbr':
            result = self.pcfg(rules, terms, lens=input['seq_len'], viterbi=False, mbr=True, depth=depth_mode)
        else:
            raise NotImplementedError

        if depth > 0:
            result['depth'] = self.part(rules, depth, mode='length', depth_output='full')
            result['depth'] = result['depth'].exp()
            
        result['partition'] -= rules['kl']
        return result
