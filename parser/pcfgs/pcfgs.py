import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from parser.pcfgs.fn import  stripe, diagonal_copy_depth, diagonal_copy_, diagonal_depth, diagonal, checkpoint

class PCFG_base(nn.Module):
    def get_depth_index(self, y, z, device=None):
        if not hasattr(self, 'index_cache'):
            self.index_cache = {}
        if y in self.index_cache:
            if z in self.index_cache[y]:
                index = self.index_cache[y][z]
            else:
                index = torch.maximum(
                    torch.arange(y).unsqueeze(1).expand(y, z),
                    torch.arange(z).unsqueeze(0).expand(y, z)
                )
                self.index_cache[y][z] = index
        else:
            index = torch.maximum(
                torch.arange(y).unsqueeze(1).expand(y, z),
                torch.arange(z).unsqueeze(0).expand(y, z)
            )
            self.index_cache[y] = {}
            self.index_cache[y][z] = index
        if device:
            return index.to(device)
        else:
            return index

    def get_depth_index_grid(self, d, device=None):
        index = []
        for i in range(1, d+1):
            tmp = self.get_depth_index(i, d+1-i)
            tmp = F.pad(tmp, (0, i-1, 0, d-i), value=d).reshape(-1)
            index.append(tmp)
        index = torch.stack(index, dim=0).reshape(-1)
        if device:
            return index.to(device)
        else:
            return index

    def get_depth_range(self, d, device=None):
        if not hasattr(self, 'range_cache'):
            self.range_cache = {}
        if d in self.range_cache:
            range = self.range_cache[d]
        else:
            min = (torch.tensor(d).log2().ceil().long() + 1).to(device)
            max = torch.tensor(d).to(device)
            range = (min, max)
            self.range_cache[d] = range
        return range

    def _inside(self):
        raise NotImplementedError

    def _inside_depth(self):
        raise NotImplementedError

    def inside(self, rules, lens, depth=False):
        if depth:
            return self._inside_depth(rules, lens)
        else:
            return self._inside(rules, lens)

    @torch.enable_grad()
    def decode(self, rules, lens, viterbi=False, mbr=False, depth=False):
        if depth:
            return self._inside_depth(rules=rules, lens=lens, viterbi=viterbi, mbr=mbr)
        else:
            return self._inside(rules=rules, lens=lens, viterbi=viterbi, mbr=mbr)


    def _get_prediction(self, logZ, span_indicator, lens, mbr=False, depth=False):
        batch, seq_len = span_indicator.shape[:2]
        prediction = [[] for _ in range(batch)]
        if depth:
            ds = span_indicator.shape[-1]
            for i in range(len(prediction)):
                prediction[i] = [[] for _ in range(ds)]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            assert logZ.requires_grad
            logZ.sum().backward()
            marginals = span_indicator.grad
            if mbr:
                if depth:
                    return self._cky_zero_order_depth(marginals.detach(), lens)
                else:
                    return self._cky_zero_order(marginals.detach(), lens)
            else:
                viterbi_spans = marginals.nonzero().tolist()
                for span in viterbi_spans:
                    if depth:
                        prediction[span[0]][span[3]].append((span[1], span[2]))
                    else:
                        prediction[span[0]].append((span[1], span[2]))
        return prediction


    @torch.no_grad()
    def _cky_zero_order(self, marginals, lens):
        N = marginals.shape[-1]
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        p = marginals.new_zeros(*marginals.shape).long()
        diagonal_copy_(s, diagonal(marginals, 1), 1)
        for w in range(2, N):
            n = N - w
            starts = p.new_tensor(range(n))
            if w != 2:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            else:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            X, split = (Y + Z).max(2)
            x = X + diagonal(marginals, w)
            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)

        def backtrack(p, i, j):
            if j == i + 1:
                return [(i, j)]
            split = p[i][j]
            ltree = backtrack(p, i, split)
            rtree = backtrack(p, split, j)
            return [(i, j)] + ltree + rtree

        p = p.tolist()
        lens = lens.tolist()
        spans = [backtrack(p[i], 0, length) for i, length in enumerate(lens)]
        return spans

    @torch.no_grad()
    def _cky_zero_order_depth(self, marginals, lens):
        batch = marginals.shape[0]
        N = marginals.shape[-1]
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        p = marginals.new_zeros(*marginals.shape).long()
        diagonal_copy_(s, diagonal(marginals, 1), 1)
        for w in range(2, N):
            min_d, max_d = self.get_depth_range(w, device=marginals.device)
            n = N - w
            starts = p.new_tensor(range(n))

            Y = stripe(s, n, w - 1, (0, 1))[..., 1:w]
            Z = stripe(s, n, w - 1, (1, w), 0)[..., 1:w]

            YZ = (Y.unsqueeze(-1) + Z.unsqueeze(-2))
            X, split = scatter_max(
                YZ.reshape(batch, n, -1),
                self.get_depth_index_grid(w-1, Y.device)
            )
            if w != 2:
                X = X[..., :-1]
                split = split[..., :-1]

            x = torch.cat([X.new_zeros(batch, n, 1), X], dim=-1) + diagonal_depth(marginals, w, (1, w))
            split = torch.cat([split.new_zeros(batch, n, 1), split], dim=-1)

            diagonal_copy_depth(s, x, w, (1, w))
            diagonal_copy_depth(p, split + YZ.stride()[2]*(starts[None, :, None] + 1), w, (1, w))

        def backtrack_depth(p, i, j, d):
            if j == i + 1:
                return [(i, j)]
            split, ld, rd = index_to_split(p[i][j][d], j-i)
            ltree = backtrack_depth(p, i, split, ld)
            rtree = backtrack_depth(p, split, j, rd)
            return [(i, j)] + ltree + rtree

        def index_to_split(i, d):
            d = d-1
            stride = (d*d, d, 1)
            split = i // stride[0]
            i = i - split*stride[0]
            ld = i // stride[1]
            i = i - ld*stride[1]
            rd = i
            return split, ld+1, rd+1

        p = p.tolist()
        lens = lens.tolist()
        spans = []
        min_d, max_d = self.get_depth_range(N-1)
        for i, length in enumerate(lens):
            spans.append([backtrack_depth(p[i], 0, length, j) for j in range(min_d, max_d+1)])

        return spans

    def get_plus_semiring(self, viterbi):
        if viterbi:
            def plus(x, dim):
                return torch.max(x, dim)[0]
        else:
            def plus(x, dim):
                return torch.logsumexp(x, dim)
        return plus

    def _eisner(self, attach, lens):
        self.huge = -1e9
        self.device = attach.device
        '''
        :param attach: The marginal probabilities.
        :param lens: sentences lens
        :return: predicted_arcs
        '''
        A = 0
        B = 1
        I = 0
        C = 1
        L = 0
        R = 1
        b, N, *_ = attach.shape
        attach.requires_grad_(True)
        alpha = [
            [
                [torch.zeros(b, N, N, device=self.device).fill_(self.huge) for _ in range(2)] for _ in range(2)
            ] for _ in range(2)
        ]
        alpha[A][C][L][:, :, 0] = 0
        alpha[B][C][L][:, :, -1] =  0
        alpha[A][C][R][:, :, 0] =  0
        alpha[B][C][R][:, :, -1] = 0
        semiring_plus = self.get_plus_semiring(viterbi=True)
        # single root.
        start_idx = 1
        for k in range(1, N-start_idx):
            f = torch.arange(start_idx, N - k), torch.arange(k+start_idx, N)
            ACL = alpha[A][C][L][:, start_idx: N - k, :k]
            ACR = alpha[A][C][R][:,  start_idx: N - k, :k]
            BCL = alpha[B][C][L][:,  start_idx+k:, N - k:]
            BCR = alpha[B][C][R][:,  start_idx+k:, N - k :]
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_l = x + attach[:, f[1], f[0]]
            alpha[A][I][L][:,  start_idx: N - k, k] = arcs_l
            alpha[B][I][L][:, k+start_idx:N, N - k - 1] = arcs_l
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_r = x + attach[:, f[0], f[1]]
            alpha[A][I][R][:, start_idx: N - k, k] = arcs_r
            alpha[B][I][R][:, k+start_idx:N, N - k - 1] = arcs_r
            AIR = alpha[A][I][R][:, start_idx: N - k, 1 : k + 1]
            BIL = alpha[B][I][L][:, k+start_idx:, N - k - 1 : N - 1]
            new = semiring_plus(ACL + BIL, dim=2)
            alpha[A][C][L][:, start_idx: N - k, k] = new
            alpha[B][C][L][:, k+start_idx:N, N - k - 1] = new
            new = semiring_plus(AIR + BCR, dim=2)
            alpha[A][C][R][:, start_idx:N-k, k] = new
            alpha[B][C][R][:, start_idx+k:N, N - k - 1] = new
        # dealing with the root.
        root_incomplete_span = alpha[A][C][L][:, 1, :N-1] + attach[:, 0, 1:]
        for k in range(1,N):
            AIR = root_incomplete_span[:, :k]
            BCR = alpha[B][C][R][:, k, N-k:]
            alpha[A][C][R][:, 0, k] = semiring_plus(AIR+BCR, dim=1)
        logZ = torch.gather(alpha[A][C][R][:, 0, :], -1, lens.unsqueeze(-1))
        arc = torch.autograd.grad(logZ.sum(), attach)[0].nonzero().tolist()
        predicted_arc = [[] for _ in range(logZ.shape[0])]
        for a in arc:
            predicted_arc[a[0]].append((a[1] - 1, a[2] -1 ))
        return predicted_arc

    @torch.enable_grad()
    def depth_partition_function_old(self, rules, lens, mode=None, span=0):
        rule = rules['rule']
        root = rules['root']
        batch, N, NT_T, _ = rule.shape
        T = NT_T - N

        D = lens.max() + span + 1
        rule = rule.reshape(batch, N, -1)
        bias = rule.new_zeros(batch, T)
        t = rule.new_zeros(batch, D, N).fill_(-1e9)

        # Shorteset parse tree depth: ROOT - NT - T - w (minimum depth 4)
        # without root and words, minimum depth 2.
        # so we can not get any probability for the smaller depth than 2
        for d in range(2, D):
            z_prev = torch.cat((t[:, d - 1], bias), dim=1)
            z = (z_prev[:, :, None] + z_prev[:, None, :]).reshape(batch, -1)
            z = (rule + z[:, None, :]).logsumexp(2)
            z = torch.clamp(z, max=0)
            t[:, d].copy_(z)

        if mode == 'full':
            r = (root.unsqueeze(1) + t).logsumexp(2)
            r = torch.cat([r.new_zeros(batch, 2).fill_(-1e9), r], dim=1)
        elif mode == 'fit':
            min_d = lens.log2().ceil().long()
            max_d = lens + span
            min_part = (root + t[torch.arange(batch), min_d]).logsumexp(1)
            max_part = (root + t[torch.arange(batch), max_d]).logsumexp(1)
            r = (max_part.exp() - min_part.exp()).log()
            return r, max_d - min_d
        else:
            r = (root + t[torch.arange(batch), lens + span]).logsumexp(1)
        return r


    @torch.enable_grad()
    def depth_partition_function(self, rules, lens, mode=None, span=0, withRootTerm=False):
        rule = rules['rule']
        root = rules['root']
        batch, NT, NT_T, _ = rule.shape
        T = NT_T - NT
        NTs = slice(0, NT)
        Ts = slice(NT, NT_T)

        D = lens.max() + span + 1
        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, -1)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, -1)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, -1)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, -1)
        bias = rule.new_zeros(batch, T)
        t = rule.new_zeros(batch, D, NT).fill_(-1e9)

        def contract(rule, y, z, dim=-1):
            r = (y[:, :, None] + z[:, None, :]).reshape(batch, -1)
            r = (rule + r[:, None, :]).logsumexp(dim)
            return r
        
        # Shorteset parse tree depth w/ root&term: ROOT - NT - T - w (minimum depth 4)
        # Shorteset parse tree depth w/o root&term: NT - T (minimum depth 2)
        # so we can not get any probability for the smaller depth than 2
        for d in range(2, D):
            if d == 2:
                # NT -> T T
                x = X_y_z.logsumexp(2)
            else:
                x = rule.new_zeros(5, batch, NT).fill_(-1e9)
                zp = t[:, d-1]
                # NT -> d-1 tree, d-1 tree
                x[0].copy_(contract(X_Y_Z, zp, zp))
                # NT -> d-1 tree, 1~d-2 tree
                x[1].copy_(contract(X_Y_z, zp, bias))
                # NT -> 1~d-2 tree, d-1 tree
                x[2].copy_(contract(X_y_Z, bias, zp))
                if d > 3:
                    st = t[:, 2:d-1].clone().logsumexp(1)
                    x[3].copy_(contract(X_Y_Z, zp, st))
                    x[4].copy_(contract(X_Y_Z, st, zp))
                x = x.logsumexp(0)
            t[:, d].copy_(x)

        if mode == 'full':
            r = (root.unsqueeze(1) + t).logsumexp(2)
            if withRootTerm:
                r = torch.cat([r.new_zeros(batch, 2).fill_(-1e9), r], dim=1)
        else:
            r = root + t.logsumexp(1)
            r = r.logsumexp(1)
        return r


    @torch.enable_grad()
    def span_partition_function(self, rules, lens, viterbi=False, mbr=False):
        terms = rules['unary']
        rule = rules['rule']
        root = rules['root']

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = rule.new_zeros(batch, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        span_indicator = rule.new_zeros(batch, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(rule):
            b_n_x = contract(rule)
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            w = Y.shape[1] - 1
            b_n_yz = (Y[:, :-1, :, None] + Z[:, 1:, None, :]).reshape(batch, w, -1).logsumexp(1)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule)
            return b_n_x

        @checkpoint
        def XYz(Y, rule):
            Y = Y[:, -1, :, None]
            b_n_yz = Y.expand(batch, NT, T).reshape(batch, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule)
            return b_n_x


        @checkpoint
        def XyZ(Z, rule):
            Z = Z[:, 0, None, :]
            b_n_yz = Z.expand(batch, T, NT).reshape(batch, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule)
            return b_n_x


        for w in range(2, N):
            if w == 2:
                s[:, w].copy_(Xyz(X_y_z) + span_indicator[:, w].unsqueeze(-1))
                continue

            x = terms.new_zeros(3, batch, NT).fill_(-1e9)

            Y = s[:, 2:w].clone()
            Z = s[:, 2:w].flip(1).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, X_Y_z))
            x[2].copy_(XyZ(Z, X_y_Z))

            s[:, w].copy_(contract(x, dim=0) + span_indicator[:, w].unsqueeze(-1))

        logZ = contract(s[torch.arange(batch), lens] + root)
        return logZ

    def _partition_function(self, rules, lens, mode='span', depth_output=None, span=0):
        if mode == 'depth':
            if type(lens) == int:
                lens = rules['root'].new_tensor([lens]).long()
            return self.depth_partition_function(rules, lens, depth_output, span)
        elif mode == 'span':
            return self.span_partition_function(rules, lens)