from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_depth, diagonal_copy_, diagonal, checkpoint
import torch
from torch_scatter import scatter_logsumexp


class PCFG(PCFG_base):

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
    def _inside_v2(self, rules, lens, viterbi=False, mbr=False):
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

        index_cache = {}
        def depth_add_(y, d):
            if d in index_cache:
                index = index_cache[d]
            else:
                index = torch.maximum(
                    torch.arange(d).unsqueeze(1).expand(d, d),
                    torch.arange(d).unsqueeze(0).expand(d, d)).reshape(-1).to(y.device)
                index_cache[d] = index
            return scatter_logsumexp(y, index)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            b_n_x = torch.cat([b_n_x.new_zeros(*b_n_x.shape, 2).fill_(-1e9), b_n_x.unsqueeze(-1)], dim=-1)
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            d = Y.shape[-1]
            b_n_yz = contract(Y[:, :, 1:-1, :, None, :, None] + Z[:, :, 1:-1, None, :, None, :], dim=2).reshape(batch, n, NT * NT, -1)
            b_n_yz = depth_add_(b_n_yz, d)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule[:, None, :, :, None], dim=-2)
            b_n_x = torch.cat([b_n_x.new_zeros(*b_n_x.shape[:-1], 1).fill_(-1e9), b_n_x], dim=-1)
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            d = Y.shape[-1]
            Y = Y[:, :, -1, :, None]
            z = z.unsqueeze(-1)
            b_n_yz = (Y + z).reshape(batch, n, NT * T, d)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule[:, None, :, :, None], dim=-2)
            b_n_x = torch.cat([b_n_x.new_zeros(*b_n_x.shape[:-1], 1).fill_(-1e9), b_n_x], dim=-1)
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            d = Z.shape[-1]
            Z = Z[:, :, 0, None, :]
            y = y.unsqueeze(-1)
            b_n_yz = (y + Z).reshape(batch, n, NT * T, d)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule[:, None, :, :, None], dim=-2)
            b_n_x = torch.cat([b_n_x.new_zeros(*b_n_x.shape[:-1], 1).fill_(-1e9), b_n_x], dim=-1)
            return b_n_x


        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1:, None, :]

            if w == 2:
                diagonal_copy_depth(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w, None, :w+1], w, w+1)
                continue

            x = terms.new_zeros(3, batch, n, NT, w+1).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1))[..., :w].clone()
            Z = stripe(s, n, w - 1, (1, w), 0)[..., :w].clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            diagonal_copy_depth(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w, None, :w+1], w, w+1)

        logZ = contract(s[torch.arange(batch), 0, lens] + root.unsqueeze(-1), dim=-2)

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            return {'partition': logZ,
                    'prediction': prediction}

        else:
            return {'partition': logZ}