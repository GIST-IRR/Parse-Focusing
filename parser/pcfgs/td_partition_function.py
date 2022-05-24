import torch
from parser.pcfgs.partition_function import PartitionFunction

class TDPartitionFunction(PartitionFunction):
    def depth_partition_function(self, rules, depth, output=None, span=0):
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

    def length_partition_function(self, rules, lens, output=None):
        unary = rules['unary']
        root = rules['root']

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules['head']  # (batch, NT, r) r:=rank
        L = rules['left']  # (batch, NT+T, r)
        R = rules['right'] # (batch, NT+T, r)

        T = unary.shape[1]
        S = L.shape[-2]
        NT = S - T

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        # @checkpoint
        def transform_left_t(left):
            '''
            :param left: shape (batch, T, r)
            :return: shape (batch, r)
            '''
            return left.logsumexp(1)

        # @checkpoint
        def transform_left_nt(x, left):
            '''
            :param x: shape (batch, NT)
            :param left: shape (batch, NT, r)
            :return: shape (batch, r)
            '''
            return (x.unsqueeze(-1) + left).logsumexp(-2)

        # @checkpoint
        def transform_right_t(right):
            return right.logsumexp(1)

        # @checkpoint
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

        left_s[:, 1].copy_(left_term)
        right_s[:, 1].copy_(right_term)

        # w: span width
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            Y = left_s[:, 1:w].clone()
            Z = right_s[:, 1:w].flip(1).clone()
            x = merge(Y, Z)
            if w + 1 < N:
                left_x = transform_left_nt(x, L_nonterm)
                right_x = transform_right_nt(x, R_nonterm)
                left_s[:, w].copy_(left_x)
                right_s[:, w].copy_(right_x)
            s[:, w].copy_(x)

        if output == 'full':
            logZ = (s + root.unsqueeze(1)).logsumexp(-1)
        else:
            logZ = (s[torch.arange(batch), lens] + root).logsumexp(-1)
        return logZ

    def length_partition_function_full(self, rules, lens, output=None):
        unary = rules['unary']
        root = rules['root']

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules['head']  # (batch, NT, r) r:=rank
        L = rules['left']  # (batch, NT+T, r)
        R = rules['right'] # (batch, NT+T, r)

        T = unary.shape[1]
        S = L.shape[-2]
        NT = S - T

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        # @checkpoint
        def transform_left_t(terms, left):
            '''
            :param terms: shape (batch, T)
            :param left: shape (batch, T, r)
            :return: shape (batch, r)
            '''
            return (terms.unsqueeze(-1) + left).logsumexp(1)

        # @checkpoint
        def transform_left_nt(x, left):
            '''
            :param x: shape (batch, NT)
            :param left: shape (batch, NT, r)
            :return: shape (batch, r)
            '''
            return (x.unsqueeze(-1) + left).logsumexp(-2)

        # @checkpoint
        def transform_right_t(terms, right):
            return (terms.unsqueeze(-1) + right).logsumexp(1)

        # @checkpoint
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

        terms = unary.logsumexp(-1)
        left_term = transform_left_t(terms, L_term)
        right_term = transform_right_t(terms, R_term)

        s = unary.new_zeros(batch, N, NT).fill_(-1e9)
        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, L.shape[2]).fill_(-1e9)

        left_s[:, 1].copy_(left_term)
        right_s[:, 1].copy_(right_term)

        # w: span width
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            Y = left_s[:, 1:w].clone()
            Z = right_s[:, 1:w].flip(1).clone()
            x = merge(Y, Z)
            if w + 1 < N:
                left_x = transform_left_nt(x, L_nonterm)
                right_x = transform_right_nt(x, R_nonterm)
                left_s[:, w].copy_(left_x)
                right_s[:, w].copy_(right_x)
            s[:, w].copy_(x)

        if output == 'full':
            logZ = (s + root.unsqueeze(1)).logsumexp(-1)
        else:
            logZ = (s[torch.arange(batch), lens] + root).logsumexp(-1)
        return logZ