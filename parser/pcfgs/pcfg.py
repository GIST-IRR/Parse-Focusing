from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_depth_old, diagonal_copy_depth, diagonal_copy_, diagonal, checkpoint
import torch
from torch_scatter import scatter_logsumexp


class PCFG(PCFG_base):
    def get_depth_index_grid(self, y, z, device=None):
        if not hasattr(self, 'index_cache'):
            self.index_cache = {}
        if y in self.index_cache:
            if z in self.index_cache[y]:
                index = self.index_cache[y][z]
            else:
                index = torch.maximum(
                    torch.arange(y).unsqueeze(1).expand(y, z),
                    torch.arange(z).unsqueeze(0).expand(y, z)
                ).reshape(-1).to(device)
                self.index_cache[y][z] = index
        else:
            index = torch.maximum(
                torch.arange(y).unsqueeze(1).expand(y, z),
                torch.arange(z).unsqueeze(0).expand(y, z)
            ).reshape(-1).to(device)
            self.index_cache[y] = {}
            self.index_cache[y][z] = index
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

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        terms = rules['unary']
        rule = rules['rule']
        root = rules['root']

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2), dim=2).reshape(batch, n, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x


        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x


        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1:, None, :]

            if w == 2:
                diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                continue

            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            return {'partition': logZ,
                    'prediction': prediction}

        else:
            return {'partition': logZ}

    @torch.enable_grad()
    def _inside_depth(self, rules, lens, viterbi=False, mbr=False):
        terms = rules['unary']
        rule = rules['rule']
        root = rules['root']

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T
        D = N

        s = terms.new_zeros(batch, N, N, NT, D).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        span_indicator = rule.new_zeros(batch, N, N, D).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            # (batch, spans, NTs, depth)
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)).unsqueeze(-1)
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule, min_d, max_d):
            _, n, w, nt, d = Y.shape
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            b_n_yz = Y.new_zeros(*Y.shape[:-2], NT*NT, max_d-min_d+1).fill_(-1e9)
            for i in range(w-2):
                min_y, max_y = self.get_depth_range(i+2, Y.device)
                min_z, max_z = self.get_depth_range(w-1-i, Z.device)
                yz = (Y[:, :, i, :, None, min_y:max_y+1, None] + Z[:, :, i, None, :, None, min_z:max_z+1]).reshape(batch, n, NT*NT, -1)
                yz = scatter_logsumexp(yz, self.get_depth_index_grid(max_y-min_y+1, max_z-min_z+1, yz.device))
                min_yz = torch.maximum(min_y, min_z) + 1
                max_yz = torch.maximum(max_y, max_z) + 1
                b_n_yz[:, :, i, :, min_yz-min_d:max_yz-max_d].copy_(yz)
            b_n_yz = contract(b_n_yz, dim=2)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule[:, None, :, :, None], dim=-2)[..., :-1]
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule, min_d, max_d):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None, min_d:max_d+1]
            z = z.unsqueeze(-1)
            b_n_yz = (Y + z).reshape(batch, n, NT * T, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule[:, None, :, :, None], dim=-2)
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule, min_d, max_d):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :, min_d:max_d+1]
            y = y.unsqueeze(-1)
            b_n_yz = (y + Z).reshape(batch, n, NT * T, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule[:, None, :, :, None], dim=-2)
            return b_n_x


        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1:, None, :]

            min_d, max_d = self.get_depth_range(w, device=terms.device)
            d_size = max_d - min_d +1
            if w == 2:
                diagonal_copy_depth(
                    s,
                    Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w, None, min_d:max_d+1],
                    w,
                    (min_d, max_d))
                continue

            x = terms.new_zeros(3, batch, n, NT, d_size).fill_(-1e9)

            p_min, p_max = self.get_depth_range(w-1, device=terms.device)
            Y = stripe(s, n, w - 1, (0, 1))[..., :w].clone()
            Z = stripe(s, n, w - 1, (1, w), 0)[..., :w].clone()

            if w > 3:
                x[0, ..., :-1].copy_(XYZ(Y, Z, X_Y_Z, min_d, max_d))

            x[1, ..., p_min-min_d+1:].copy_(XYz(Y, Z_term, X_Y_z, p_min, p_max))
            x[2, ..., p_min-min_d+1:].copy_(XyZ(Y_term, Z, X_y_Z, p_min, p_max))

            diagonal_copy_depth(
                s,
                contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w, None, min_d:max_d+1],
                w,
                (min_d, max_d))

        logZ = contract(s[torch.arange(batch), 0, lens] + root.unsqueeze(-1), dim=-2)

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr, depth=True)
            return {'partition': logZ,
                    'prediction': prediction}

        else:
            return {'partition': logZ}