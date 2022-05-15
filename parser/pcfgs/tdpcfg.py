from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import  stripe, diagonal_copy_, checkpoint

import torch

class TDPCFG(PCFG_base):
    def __init__(self):
        super(TDPCFG, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False):
        assert viterbi is not True
        unary = rules['unary']
        root = rules['root']

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules['head']  # (batch, NT, r) r:=rank
        L = rules['left']  # (batch, NT+T, r)
        R = rules['right'] # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        @checkpoint
        def transform_left_t(x, left):
            '''
            :param x: shape (batch, n, T)
            :return: shape (batch, n, r)
            '''
            return (x.unsqueeze(-1) + left.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_left_nt(x, left):
            '''
            :param x: shape (batch, n, NT)
            :param left: shape (batch, NT, r)
            :return: shape (batch, n, r)
            '''
            return (x.unsqueeze(-1) + left.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_right_t(x, right):
            return (x.unsqueeze(-1) + right.unsqueeze(1)).logsumexp(2)

        @checkpoint
        def transform_right_nt(x, right):
            return (x.unsqueeze(-1) + right.unsqueeze(1)).logsumexp(2)

        # @checkpoint
        def merge(Y, Z):
            '''
            :param Y: shape (batch, n, w, r)
            :param Z: shape (batch, n, w, r)
            :return: shape (batch, n, x)
            '''
            # contract dimension w.
            b_n_r = (Y + Z).logsumexp(-2)
            # contract dimension r.
            b_n_x = (b_n_r.unsqueeze(-2) + H.unsqueeze(1)).logsumexp(-1)
            return b_n_x


        batch, N, *_ = unary.shape
        N += 1

        # for estimating marginals.
        span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)

        left_term = transform_left_t(unary,L_term)
        right_term = transform_right_t(unary,R_term)

        s = unary.new_zeros(batch, N, N, NT).fill_(-1e9)
        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)

        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # w: span width
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            x = merge(Y.clone(), Z.clone())
            x = x + span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(-1)
            if w + 1 < N:
                left_x = transform_left_nt(x,L_nonterm)
                right_x = transform_right_nt(x, R_nonterm)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
            diagonal_copy_(s, x, w)

        final = s[torch.arange(batch), 0, lens] + root
        logZ = final.logsumexp(-1)

        if not mbr and not viterbi:
            return {'partition': logZ}

        else:

            return {
                    "prediction" : self._get_prediction(logZ, span_indicator, lens, mbr=True),
                    "partition" : logZ
                    }
    
    def _compose(self, rules):
        parent = rules['head']
        left = rules['left']
        right = rules['right']
        b, nt, r = parent.shape
        t = left.shape[1]
        result = \
            (parent.view(b, nt, 1, 1, r) \
            + left.view(b, 1, t, 1, r) \
            + right.view(b, 1, 1, t, r)).logsumexp(dim=-1)
        return result

    def t_partition_function(self, rules, depth, mode):
        composed = self._compose(rules)
        rules = {
            'root': rules['root'],
            'rule': composed,
            'unary': rules['unary']
        }
        return super()._partition_function(rules, depth, mode=mode)

    @torch.enable_grad()
    def depth_partition_function(self, rules, depth):
        eps = 1e-8
        terms = rules['unary']
        root = rules['root']
        H = rules['head']
        L = rules['left']
        R = rules['right']
        batch_size, NT, r = H.shape
        N = L.shape[1]
        T = N - NT

        bias = torch.zeros(batch_size, T).cuda()
        t = torch.zeros(batch_size, NT).fill_(-1e9).cuda()

        # in case of sentence depth: NT - T - w (minimum depth 3)
        # in case of parse tree depth: S - NT - T (minimum depth 3)
        # so we can not get any probability for the smaller depth than 3
        # the partition number depth is based on parse tree depth
        def logmatmul(x, y, dim=-1):
            return (x + y).logsumexp(dim=dim)

        if depth > 2:
            for _ in range(depth - 2):
                t = torch.cat((t, bias), 1)
                t = logmatmul(L, t.unsqueeze(-1), dim=1) + logmatmul(R, t.unsqueeze(-1), dim=1)
                t = logmatmul(H, t.unsqueeze(1))
                t = torch.clamp(t, max=0)

        r = torch.logsumexp(root + t.squeeze(), dim=1, keepdim=True)
        return r.squeeze(1)


    @torch.enable_grad()
    def length_partition_function(self, rules, lens, output=None):
        unary = rules['unary']
        root = rules['root']

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules['head']  # (batch, NT, r) r:=rank
        L = rules['left']  # (batch, NT+T, r)
        R = rules['right'] # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        @checkpoint
        def transform_left_t(left):
            '''
            :param left: shape (batch, T, r)
            :return: shape (batch, r)
            '''
            return left.logsumexp(1)

        @checkpoint
        def transform_left_nt(x, left):
            '''
            :param x: shape (batch, NT)
            :param left: shape (batch, NT, r)
            :return: shape (batch, r)
            '''
            return (x.unsqueeze(-1) + left).logsumexp(-2)

        @checkpoint
        def transform_right_t(right):
            return right.logsumexp(1)

        @checkpoint
        def transform_right_nt(x, right):
            return (x.unsqueeze(-1) + right).logsumexp(-2)

        # @checkpoint
        def merge(Y, Z):
            '''
            :param Y: shape (batch, w, r)
            :param Z: shape (batch, w, r)
            :return: shape (batch, NT)
            '''
            # contract dimension w.
            b_r = (Y + Z).logsumexp(-2)
            # contract dimension r.
            b_x = (b_r.unsqueeze(1) + H).logsumexp(-1)
            return b_x


        batch, *_ = unary.shape
        N = lens.max()
        N += 1

        left_term = transform_left_t(L_term)
        right_term = transform_right_t(R_term)

        s = unary.new_zeros(batch, N, NT).fill_(-1e9)
        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, L.shape[2]).fill_(-1e9)

        # diagonal_copy_(left_s, left_term, w=1)
        # diagonal_copy_(right_s, right_term, w=1)
        left_s[:, 1].copy_(left_term)
        right_s[:, 1].copy_(right_term)

        # w: span width
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            # Y = stripe(left_s, n, w - 1, (0, 1))
            # Z = stripe(right_s, n, w - 1, (1, w), 0)
            Y = left_s[:, 1:w].clone()
            Z = right_s[:, 1:w].flip(1).clone()
            x = merge(Y, Z)
            if w + 1 < N:
                left_x = transform_left_nt(x, L_nonterm)
                right_x = transform_right_nt(x, R_nonterm)
                # diagonal_copy_(left_s, left_x, w)
                # diagonal_copy_(right_s, right_x, w)
                left_s[:, w].copy_(left_x)
                right_s[:, w].copy_(right_x)
            # diagonal_copy_(s, x, w)
            s[:, w].copy_(x)

        # final = s[torch.arange(batch), lens] + root
        # logZ = final.logsumexp(-1)

        if output == 'full':
            logZ = (s + root.unsqueeze(1)).logsumexp(-1)
        else:
            logZ = (s[torch.arange(batch), lens] + root).logsumexp(-1)
        return logZ