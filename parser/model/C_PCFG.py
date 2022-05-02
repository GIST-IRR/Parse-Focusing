import torch
import torch.distributions as dist
from torch import dsmm
import torch.nn as nn
from parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from ..pcfgs.pcfg import PCFG
from torch.profiler import record_function

class CompoundPCFG(nn.Module):
    def __init__(self, args, dataset):
        super(CompoundPCFG, self).__init__()

        self.pcfg = PCFG()
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
        if hasattr(args, 'fix_root'):
            self.root_emb.requires_grad = args.fix_root
        if hasattr(args, 'fix_nonterm'):
            self.nonterm_emb.requires_grad = args.fix_nonterm
        if hasattr(args, 'fix_term'):
            self.term_emb.requires_grad = args.fix_term

        if hasattr(args, 'fix_root_mlp') and args.fix_root_mlp:
            for p in self.root_mlp.parameters():
                p.requires_grad = False
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_switch(self, switch: bool):
        self.switch = switch

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
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
            z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
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

        return {'unary': unary,
                'root': root,
                'rule': rule,
                'kl': kl(mean, lvar).sum(1)}

    def loss(self, input, partition=False, max_depth=0):
        self.partition = partition
        # with record_function("rule_setup"):
        rules = self.forward(input)
        self.rules = rules
        # Partition function
        if partition:
            # self.pf = self.pcfg._partition_function(rules=rules, lens=input['seq_len'], mode=self.mode, depth_output='fit')
            # result['partition'] = result['partition'] - self.pf  # normal

            # depth-conditioned inside algorithm
            max_d = input['seq_len'].max()
            min_d = input['seq_len'].min().log2().ceil().long().item() + 1
            # with record_function("inside_algorithm"):
            # result = self.pcfg.inside(rules=rules, lens=input['seq_len'], depth=True)
            # result['partition'] = result['partition'][:, min_d:max_d+1]

            result = self.pcfg.inside(rules=rules, lens=input['seq_len'], depth=False)

            # partition function approximation
            # with record_function("PF_approximation"):
            if max_depth == 0:
                lens = input['seq_len']
            else:
                lens = max_depth
            # self.pf = self.pcfg._partition_function(rules=rules, lens=input['seq_len'], mode=self.mode, depth_output='full')
            self.pf = self.pcfg._partition_function(rules=rules, lens=lens, mode=self.mode, depth_output='full')
            # remain = torch.cat([self.pf[:, 2:min_d], self.pf[:, max_d+1:]], dim=1)
            self.pf = self.pf[:, min_d:max_d+1]

            # Entropy Calculation
            # tau = 0.1 
            log_p_d = result['partition']
            # log_p_d = result['partition'] - result['partition'].logsumexp(-1).unsqueeze(-1) # p(d|w, G)
            # cat = dist.categorical.Categorical(logits=log_p_d)
            # self.ent = cat.entropy()
            # ent = torch.where(self.ent > tau * self.ent.new_tensor([max_d-min_d+1]).log(), self.ent, -self.ent)

            # PF entropy
            # log_p_d = self.pf - self.pf.logsumexp(-1).unsqueeze(-1)
            # cat = dist.categorical.Categorical(logits=log_p_d)
            # ent = cat.entropy()

            # Renormalization
            result['partition'] = result['partition'] - self.pf.logsumexp(-1)
            # result['partition'] = (result['partition'] - self.pf).logsumexp(-1)
            # result['partition'] = (result['partition'] - self.pf).sum(-1) + ent  # maximization = to uniform
            # result['partition'] = (result['partition'] - self.pf).sum(-1) - self.ent  # minimization = to one-hot
        else:
            # result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])

            result = self.pcfg.inside(rules=rules, lens=input['seq_len'], depth=False)
            # min_d = input['seq_len'].min().log2().ceil().long().item() + 1
            # result['partition'] = result['partition'][:, min_d:]

            log_p_d = result['partition']
            # log_p_d = result['partition'] - result['partition'].logsumexp(-1).unsqueeze(-1) # p(d|w, G)
            # cat = dist.categorical.Categorical(logits=log_p_d)
            # self.ent = cat.entropy()

            # result['partition'] = result['partition'].logsumexp(-1)

        # depth-conditioned inside algorithm
        loss =  (-result['partition'] + rules['kl']).mean()
        return loss, log_p_d


    def evaluate(self, input, decode_type, depth=0, **kwargs):
        if not hasattr(self, 'partition'):
            self.partition = False
        # partition = self.partition
        partition = False
        # partition = True

        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=True, mbr=False, depth=partition)
        elif decode_type == 'mbr':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True, depth=partition)
        else:
            raise NotImplementedError

        if depth > 0:
            # max_depth = max_d if depth < max_d else depth
            result['depth'] = self.pcfg._partition_function(rules, depth, mode='depth', depth_output='full')
            if partition:
                min_d = input['seq_len'].log2().ceil().long().min() + 1
                max_d = input['seq_len'].max()
                batch, _ = result['partition'].shape
                result['partition'] = (result['partition'][:, min_d:max_d+1] - result['depth'][:, min_d:max_d+1])
                idx = result['partition'].argmax(-1)

                result['partition'] = result['partition'][torch.arange(batch), idx]

                # result['prediction_o'] = [result['prediction'][i][tmp_idx[i]] for i in range(batch)] # original prediction without normal.
                # result['prediction'] = [result['prediction'][i][torch.randint(0, max_d-min_d+1, (1,))] for i in range(batch)] # random prediction
                # result['prediction'] = [result['prediction'][i][(torch.randn(1)+((max_d-min_d)/2).cpu()).clamp(0, max_d-min_d).round().long()] for i in range(batch)]
                # result['prediction_o'] = _result['prediction']
                # result['prediction'] = [result['prediction'][i][idx[i]] for i in range(batch)]

            result['depth'] = result['depth'].exp()
            
        result['partition'] -= rules['kl']
        return result
