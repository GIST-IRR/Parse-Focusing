import torch
import torch.nn as nn
from torch import Tensor


class PartitionFunction(nn.Module):
    def depth_partition_function(
        self,
        rules: dict,
        lens: Tensor,
        mode: str = None,
        span: int = 0,
        with_root_term: bool = False,
        until_converge: bool = False,
    ):
        rule = rules["rule"]
        root = rules["root"]
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
                zp = t[:, d - 1]
                # NT -> d-1 tree, d-1 tree
                x[0].copy_(contract(X_Y_Z, zp, zp))
                # NT -> d-1 tree, 1 tree
                x[1].copy_(contract(X_Y_z, zp, bias))
                # NT -> 1 tree, d-1 tree
                x[2].copy_(contract(X_y_Z, bias, zp))
                if d > 3:
                    st = t[:, 2 : d - 1].clone().logsumexp(1)
                    # NT -> d-1 tree, d-2~2 tree
                    x[3].copy_(contract(X_Y_Z, zp, st))
                    # NT -> d-2~2 tree, d-1 tree
                    x[4].copy_(contract(X_Y_Z, st, zp))
                x = x.logsumexp(0)
            t[:, d].copy_(x)

            if until_converge and (
                t[:, :d].logsumexp(1) == t[:, :d+1].logsumexp(1)).all():
                break

        if mode == "full":
            r = (root.unsqueeze(1) + t).logsumexp(2)
            if with_root_term:
                r = torch.cat([r.new_zeros(batch, 2).fill_(-1e9), r], dim=1)
        else:
            r = root + t.logsumexp(1)
            r = r.logsumexp(1)
        return r

    def length_partition_function(self, rules, lens, mode=None):
        root = rules["root"]
        rule = rules["rule"]

        batch, NT, S, _ = rule.shape
        T = S - NT
        N = lens.max() + 1

        s = rule.new_zeros(batch, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        def contract(x, dim=-1):
            return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        # @checkpoint
        def Xyz(rule):
            # y_z = (term[:, :, None] + term[:, None, :]).reshape(batch, T * T)
            b_n_x = contract(rule)
            return b_n_x

        # @checkpoint
        def XYZ(Y, Z, rule):
            w = Y.shape[1] - 1
            b_n_yz = (
                (Y[:, :-1, :, None] + Z[:, 1:, None, :])
                .reshape(batch, w, -1)
                .logsumexp(1)
            )
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule)
            return b_n_x

        # @checkpoint
        def XYz(Y, rule):
            Y = Y[:, -1, :, None]
            b_n_yz = Y.expand(batch, NT, T).reshape(batch, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule)
            return b_n_x

        # @checkpoint
        def XyZ(Z, rule):
            Z = Z[:, 0, None, :]
            b_n_yz = Z.expand(batch, T, NT).reshape(batch, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule)
            return b_n_x

        for w in range(2, N):
            if w == 2:
                # s[:, w].copy_(Xyz(X_y_z) + span_indicator[:, w].unsqueeze(-1))
                s[:, w].copy_(Xyz(X_y_z))
                continue

            x = rule.new_zeros(3, batch, NT).fill_(-1e9)

            Y = s[:, 2:w].clone()
            Z = s[:, 2:w].flip(1).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, X_Y_z))
            x[2].copy_(XyZ(Z, X_y_Z))

            # s[:, w].copy_(contract(x, dim=0) + span_indicator[:, w].unsqueeze(-1))
            s[:, w].copy_(contract(x, dim=0))

        if mode == "full":
            logZ = contract(s + root.unsqueeze(1))
        else:
            logZ = contract(s[torch.arange(batch), lens] + root)
        return logZ

    def length_partition_function_full(
        self, rules: dict, lens: Tensor, mode: str = None
    ) -> Tensor:
        word = rules["word"]
        root = rules["root"]
        rule = rules["rule"]
        term = rules["unary"]

        batch, NT, S, _ = rule.shape
        T = S - NT
        N = lens.max() + 1

        s = rule.new_zeros(batch, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        def contract(x, dim=-1):
            return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        # @checkpoint
        def Xyz(rule, term):
            y_z = (term[:, :, None] + term[:, None, :]).reshape(batch, 1, T * T)
            b_n_x = contract(rule + y_z)
            return b_n_x

        # @checkpoint
        def XYZ(rule, Y, Z):
            w = Y.shape[1] - 1
            b_n_yz = (
                (Y[:, :-1, :, None] + Z[:, 1:, None, :])
                .reshape(batch, w, -1)
                .logsumexp(1)
            )
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule)
            return b_n_x

        # @checkpoint
        def XYz(rule, Y, term):
            Yz = Y[:, -1, :, None] + term[:, None, :]
            b_n_yz = Yz.expand(batch, NT, T).reshape(batch, 1, NT * T)
            b_n_x = contract(b_n_yz + rule)
            return b_n_x

        # @checkpoint
        def XyZ(rule, term, Z):
            yZ = term[:, :, None] + Z[:, 0, None, :]
            b_n_yz = yZ.expand(batch, T, NT).reshape(batch, 1, NT * T)
            b_n_x = contract(b_n_yz + rule)
            return b_n_x

        def unique_sum():
            b = term.shape[0]
            u_word = [w.unique() for w in word]
            result = []
            for i, u in enumerate(u_word):
                t = term[i, :, u]
                t = t.logsumexp(-1)
                result.append(t)
            return torch.stack(result)

        # y_z = term.logsumexp(-1)
        # y_z = term.logsumexp(1)
        y_z = unique_sum()
        for w in range(2, N):
            if w == 2:
                s[:, w].copy_(Xyz(X_y_z, y_z))
                continue

            x = rule.new_zeros(3, batch, NT).fill_(-1e9)

            Y = s[:, 2:w].clone()
            Z = s[:, 2:w].flip(1).clone()

            if w > 3:
                x[0].copy_(XYZ(X_Y_Z, Y, Z))

            x[1].copy_(XYz(X_Y_z, Y, y_z))
            x[2].copy_(XyZ(X_y_Z, y_z, Z))

            s[:, w].copy_(contract(x, dim=0))

        if mode == "full":
            logZ = contract(s + root.unsqueeze(1))
        else:
            logZ = contract(s[torch.arange(batch), lens] + root)
        return logZ

    def forward(self, rules, lens, mode="length", depth_output=None, span=0, until_converge=False):
        if type(lens) == int:
            lens = rules["root"].new_tensor([lens]).long()
        if mode == "depth":
            return self.depth_partition_function(rules, lens, depth_output, span, until_converge=until_converge)
        elif mode == "length":
            return self.length_partition_function(rules, lens, depth_output)
        elif mode == "length_unary":
            return self.length_partition_function_full(rules, lens, depth_output)
