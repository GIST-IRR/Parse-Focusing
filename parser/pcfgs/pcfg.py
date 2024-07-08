from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import (
    stripe,
    diagonal_copy_,
    checkpoint,
)
import torch


class PCFG(PCFG_base):
    @torch.enable_grad()
    def _inside(
        self,
        rules,
        terms,
        lens,
        viterbi=False,
        mbr=False,
        dropout=0.0,
        label=False,
    ):
        # terms = rules['unary']
        rule = rules["rule"]
        root = rules["root"]

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

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        # span_indicator = rule.new_ones(batch, N, N, NT).requires_grad_(
        #     viterbi or mbr
        # )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)
                # orig_sum = x.logsumexp(dim, keepdims=True)
                # prop = (x/2).log_softmax(dim)
                # return (prop + orig_sum).logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))

            # b_n_yz = (y * z).reshape(batch, n, T * T)
            # b_n_x = (b_n_yz.unsqueeze(-2) * rule.unsqueeze(1)).sum(-1)

            # b_n_x = F.dropout(b_n_x, p=dropout, training=self.training)
            # b_n_x = b_n_x * dropout if self.training else b_n_x
            # b_n_x = torch.where(b_n_x < 0, b_n_x, -1e+9)
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(
                Y[:, :, 1:-1, :].unsqueeze(-1)
                + Z[:, :, 1:-1, :].unsqueeze(-2),
                dim=2,
            ).reshape(batch, n, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))

            # b_n_x = F.dropout(b_n_x, p=dropout, training=self.training)
            # b_n_x = b_n_x * dropout if self.training else b_n_x
            # b_n_x = torch.where(b_n_x < 0, b_n_x, -1e+9)
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))

            # b_n_x = F.dropout(b_n_x, p=dropout, training=self.training)
            # b_n_x = b_n_x * dropout if self.training else b_n_x
            # b_n_x = torch.where(b_n_x < 0, b_n_x, -1e+9)
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))

            # b_n_x = F.dropout(b_n_x, p=dropout, training=self.training)
            # b_n_x = b_n_x * dropout if self.training else b_n_x
            # b_n_x = torch.where(b_n_x < 0, b_n_x, -1e+9)
            return b_n_x

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )
                continue

            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)
        # logZ = (s[torch.arange(batch), 0, lens] * root).sum(-1)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ,
                span_indicator,
                lens,
                tag_indicator=tag_indicator,
                mbr=mbr,
                label=label,
            )
            return {"partition": logZ, "prediction": prediction}
            # return {"partition": logZ}
        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_topk(
        self, rules, terms, lens, viterbi=False, mbr=False, topk=1
    ):
        # terms = rules['unary']
        rule = rules["rule"]
        root = rules["root"]

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

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

        def contract(x, topk=None, dim=-1):
            if topk:
                dim_size = x.shape[dim]
                if dim_size < topk:
                    return x.topk(dim_size, dim)[0]
                else:
                    return x.topk(topk, dim)[0]
            else:
                return x.max(dim)[0]

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1), topk)
            b_n_x = b_n_x.logsumexp(-1)
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            l = Y.shape[2]
            b_n_yz = (Y[:, :, 1:-1, :, None] + Z[:, :, 1:-1, None, :]).reshape(
                batch, n, l - 2, -1
            )
            # b_n_x: b, n, l-2, None, NT*NT
            # rule: b, None, None, NT, NT*NT
            b_n_x = b_n_yz[:, :, :, None, :] + rule[:, None, None, :, :]
            b_n_x = b_n_x.permute(0, 1, 3, 2, 4)
            b_n_x = contract(b_n_x.reshape(batch, n, NT, -1), topk)
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1), topk)
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1), topk)
            return b_n_x

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )
                continue

            x = terms.new_zeros(3, batch, n, NT, topk).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))
            x = x.permute(1, 2, 3, 0, 4).reshape(batch, n, NT, -1)

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, topk).logsumexp(-1)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root, topk)
        logZ = logZ.logsumexp(-1)

        if viterbi or mbr:
            # prediction = self._get_prediction(
            #     logZ, span_indicator, lens, tag_indicator, mbr=mbr
            # )
            # return {"partition": logZ, "prediction": prediction}
            return {"partition": logZ}
        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_topk_trees(
        self, rules, terms, lens, viterbi=False, mbr=False, topk=1
    ):
        # Calculate only the top-k trees
        # terms = rules['unary']
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT, topk).fill_(-1e9)

        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        # span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
        #     viterbi or mbr
        # )
        # tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
        #     viterbi or mbr
        # )

        def contract(x, topk=None, dim=-1):
            if topk:
                dim_size = x.shape[dim]
                if dim_size < topk:
                    return x.topk(dim_size, dim)[0]
                else:
                    return x.topk(topk, dim)[0]
            else:
                return x.max(dim)[0]

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1), topk)
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            l = Y.shape[2]
            b_n_yz = (
                Y[:, :, 1:-1, :, None, :, None]
                + Z[:, :, 1:-1, None, :, None, :]
            ).reshape(batch, n, l - 2, NT * NT, topk * topk)
            # b_n_x: b, n, None, l-2, NT*NT
            # rule: b, None, NT, None, NT*NT
            b_n_x = (
                b_n_yz[:, :, None, :, :, :] + rule[:, None, :, None, :, None]
            )
            b_n_x = contract(b_n_x.reshape(batch, n, NT, -1), topk)
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            # Y: b, n, l, NT, topk
            # z: b, n, None, T
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None, :]
            b_n_yz = (Y + z[..., None]).reshape(batch, n, NT * T, topk)
            b_n_x = b_n_yz[:, :, None, :, :] + rule[:, None, :, :, None]
            b_n_x = contract(b_n_x.reshape(batch, n, NT, -1), topk)
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y[..., None] + Z).reshape(batch, n, NT * T, topk)
            b_n_x = b_n_yz[:, :, None, :, :] + rule[:, None, :, :, None]
            b_n_x = contract(b_n_x.reshape(batch, n, NT, -1), topk)
            return b_n_x

        # terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z),
                    # + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )
                continue

            x = terms.new_zeros(batch, n, NT, topk, 3).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[..., 0].copy_(XYZ(Y, Z, X_Y_Z))

            x[..., 1].copy_(XYz(Y, Z_term, X_Y_z))
            x[..., 2].copy_(XyZ(Y_term, Z, X_y_Z))
            x = x.reshape(batch, n, NT, -1)

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, topk),
                # + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                w,
            )

        logZ = s[torch.arange(batch), 0, lens] + root[..., None]
        logZ = contract(logZ.reshape(batch, -1), topk)
        logZ = logZ.logsumexp(-1)

        if viterbi or mbr:
            # prediction = self._get_prediction(
            #     logZ, span_indicator, lens, tag_indicator, mbr=mbr
            # )
            # return {"partition": logZ, "prediction": prediction}
            return {"partition": logZ}
        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_weighted(
        self,
        rules,
        terms,
        lens,
        C2N=None,
        w2T=None,
        viterbi=False,
        mbr=False,
        dropout=0.0,
    ):
        # terms = rules['unary']
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)

        NTs = slice(0, NT)
        Ts = slice(NT, S)

        # rule = rule.masked_fill(~C2N.bool(), -1e9)
        rule = rule + C2N if C2N is not None else rule
        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

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
            # b_n_x = b_n_yz.unsqueeze(-2) + (yzX * rule).unsqueeze(1)
            b_n_x = b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)
            # b_n_x = yzX.unsqueeze(1) * b_n_x
            b_n_x = contract(b_n_x)
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(
                Y[:, :, 1:-1, :].unsqueeze(-1)
                + Z[:, :, 1:-1, :].unsqueeze(-2),
                dim=2,
            ).reshape(batch, n, -1)
            # b_n_x = b_n_yz.unsqueeze(2) + (YZX * rule).unsqueeze(1)
            b_n_x = b_n_yz.unsqueeze(2) + rule.unsqueeze(1)
            # b_n_x = YZX.unsqueeze(1) * b_n_x
            b_n_x = contract(b_n_x)
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            # b_n_x = b_n_yz.unsqueeze(-2) + (YzX * rule).unsqueeze(1)
            b_n_x = b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)
            # b_n_x = YzX.unsqueeze(1) * b_n_x
            b_n_x = contract(b_n_x)
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            # b_n_x = b_n_yz.unsqueeze(-2) + (yZX * rule).unsqueeze(1)
            b_n_x = b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)
            # b_n_x = yZX.unsqueeze(1) * b_n_x
            b_n_x = contract(b_n_x)
            return b_n_x

        terms = terms + w2T if w2T is not None else terms
        # terms = terms.masked_fill(~w2T.bool(), -1e9)
        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )
                continue

            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ, span_indicator, lens, tag_indicator, mbr=mbr
            )
            return {"partition": logZ, "prediction": prediction}
            # return {"partition": logZ}
        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_one(
        self,
        rules,
        terms,
        lens,
        tree=None,
        viterbi=False,
        mbr=False,
        dropout=0.0,
    ):
        # terms = rules['unary']
        rule = rules["rule"]
        root = rules["root"]

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

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

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
            b_n_yz = contract(
                Y[:, :, 1:-1, :].unsqueeze(-1)
                + Z[:, :, 1:-1, :].unsqueeze(-2),
                dim=2,
            ).reshape(batch, n, -1)
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

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            # width_tree = [[s for s in b if s.diff() == w] for b in tree]
            width_idx = terms.new_ones(batch * n).bool()
            ts = [
                s[0].item() + i * n
                for i, b in enumerate(tree)
                for s in b
                if s.diff() == w
            ]
            width_idx[ts] = False
            width_idx = width_idx.reshape(batch, n, 1, 1)

            Y_term = Y_term.masked_fill(width_idx, -1e9)
            Z_term = Z_term.masked_fill(width_idx, -1e9)

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )
                continue

            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()
            Y = Y.masked_fill(width_idx, -1e9)
            Z = Z.masked_fill(width_idx, -1e9)

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ, span_indicator, lens, tag_indicator, mbr=mbr
            )
            return {"partition": logZ, "prediction": prediction}
            # return {"partition": logZ}
        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_one_new(
        self,
        rules,
        terms,
        lens,
        tree=None,
        viterbi=False,
        mbr=False,
        dropout=0.0,
    ):
        # terms = rules['unary']
        rule = rules["rule"]
        root = rules["root"]

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

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        def mask_unselect(x, n, mask):
            ret = x.new_full([batch, n, *x.shape[1:]], -1e9)
            # mask = [(m//n, m%n) for m in mask]
            tmp = [[] for _ in range(batch)]
            for m in mask:
                tmp[m // n].append(m % n)
            mask = tmp

            start = 0
            for i, m in enumerate(mask):
                last = start + len(m)
                ret[i, m] = x[start:last]
                start = last

            return ret

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            bn = y.shape[0]
            b_n_yz = (y + z).reshape(bn, 1, T * T)
            b_n_x = contract(b_n_yz + rule[0].unsqueeze(0))

            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            bn = Y.shape[0]
            b_n_yz = contract(
                Y[:, 1:-1, :, None] + Z[:, 1:-1, None, :],
                dim=1,
            ).reshape(bn, 1, -1)
            b_n_x = contract(b_n_yz + rule[0].unsqueeze(0))

            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            bn = Y.shape[0]
            Y = Y[:, -1, :, None]
            b_n_yz = (Y + z).reshape(bn, 1, NT * T)
            b_n_x = contract(b_n_yz + rule[0].unsqueeze(0))

            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            bn = Z.shape[0]
            Z = Z[:, 0, None, :]
            b_n_yz = (y + Z).reshape(bn, 1, NT * T)
            b_n_x = contract(b_n_yz + rule[0].unsqueeze(0))

            return b_n_x

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            # Get index
            ts = [
                s[0].item() + i * n
                for i, b in enumerate(tree)
                for s in b
                if s.diff() == w
            ]
            if len(ts) == 0:
                continue

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]
            Y_term = Y_term.reshape(-1, T, 1)[ts]
            Z_term = Z_term.reshape(-1, 1, T)[ts]

            # yt = mask_unselect(Y_term, batch, n, ts)
            start = torch.arange(n)
            end = torch.arange(n) + w

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    mask_unselect(
                        Xyz(Y_term, Z_term, X_y_z)
                        + span_indicator[:, start, end].reshape(-1, NT)[ts],
                        n,
                        ts,
                    ),
                    w,
                )
                continue

            # x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)
            x = terms.new_zeros(3, len(ts), NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()
            Y = Y.reshape(-1, *Y.shape[2:])[ts]
            Z = Z.reshape(-1, *Z.shape[2:])[ts]

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                mask_unselect(
                    contract(x, dim=0)
                    + span_indicator[:, start, end].reshape(-1, NT)[ts],
                    n,
                    ts,
                ),
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ, span_indicator, lens, tag_indicator, mbr=mbr
            )
            return {"partition": logZ, "prediction": prediction}
            # return {"partition": logZ}
        else:
            return {"partition": logZ}


class Faster_PCFG(PCFG_base):
    @torch.enable_grad()
    def _inside(
        self,
        rules,
        terms,
        lens,
        dropout=None,
        tree=None,
        viterbi=False,
        mbr=False,
        label=False,
    ):
        assert viterbi == False

        # terms = rules['unary']
        rule = rules["rule"]
        # filter_mask = torch.where(
        #     rule[0].exp() < rule[0].exp().mean(), 1, 0).all(0)
        # filter_mask = torch.where(
        #     rule[0].exp() < 1e-1, 1, 0).all(0)
        # rule.masked_fill_(filter_mask[None, None, :], -1e9)
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        rule = rule.exp()
        X_Y_Z = rule[:, :, NTs, NTs].contiguous()
        X_y_Z = rule[:, :, Ts, NTs].contiguous()
        X_Y_z = rule[:, :, NTs, Ts].contiguous()
        X_y_z = rule[:, :, Ts, Ts].contiguous()

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ
        @checkpoint
        def Xyz(y, z, rule):
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            x = torch.einsum("bny, bnz, bxyz -> bnx", y, z, rule)
            x = (
                (x + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return x

        @checkpoint
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y, Z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            X = X.logsumexp(2)
            return X

        @checkpoint
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz->bnx", Y, z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return X

        @checkpoint
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz-> bnx", y, Z, rule)
            X = (
                (X + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            return X

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[
                :,
                :n,
                :,
            ]
            Z_term = terms[:, w - 1 :, :]

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[
                        :, torch.arange(n), torch.arange(n) + w, :
                    ],
                    w,
                )
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w, :],
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ,
                span_indicator,
                lens,
                mbr=mbr,
                label=label,
                tag_indicator=tag_indicator,
            )
            return {"partition": logZ, "prediction": prediction}

        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_one(
        self,
        rules,
        terms,
        lens,
        dropout=None,
        tree=None,
        viterbi=False,
        mbr=False,
    ):
        assert viterbi == False

        # terms = rules['unary']
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        rule = rule.exp()
        X_Y_Z = rule[:, :, NTs, NTs].contiguous()
        X_y_Z = rule[:, :, Ts, NTs].contiguous()
        X_Y_z = rule[:, :, NTs, Ts].contiguous()
        X_y_z = rule[:, :, Ts, Ts].contiguous()

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(
            viterbi or mbr
        )
        span_mask = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ
        @checkpoint
        def Xyz(y, z, rule):
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            x = torch.einsum("bny, bnz, bxyz -> bnx", y, z, rule)
            x = (
                (x + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return x

        @checkpoint
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y, Z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            X = X.logsumexp(2)
            return X

        @checkpoint
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz->bnx", Y, z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return X

        @checkpoint
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz-> bnx", y, Z, rule)
            X = (
                (X + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            return X

        for w in range(2, N):
            n = N - w

            # Get Mask
            if tree is not None:
                child_mask = (
                    tree[:, torch.arange(n), torch.arange(n) + w] + 1e-9
                )
                child_mask = child_mask.log()
                if child_mask.dim() == 2:
                    child_mask = child_mask[..., None]

            Y_term = terms[:, :n, :]
            Z_term = terms[:, w - 1 :, :]

            if w == 2:
                if tree is not None:
                    diagonal_copy_(
                        s,
                        (Xyz(Y_term, Z_term, X_y_z) + child_mask)
                        + span_indicator[
                            :, torch.arange(n), torch.arange(n) + w
                        ].unsqueeze(-1),
                        w,
                    )
                else:
                    diagonal_copy_(
                        s,
                        Xyz(Y_term, Z_term, X_y_z)
                        + span_indicator[
                            :, torch.arange(n), torch.arange(n) + w
                        ].unsqueeze(-1),
                        w,
                    )
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            if tree is not None:
                diagonal_copy_(
                    s,
                    (contract(x, dim=0) + child_mask)
                    + span_indicator[
                        :, torch.arange(n), torch.arange(n) + w
                    ].unsqueeze(-1),
                    w,
                )
            else:
                diagonal_copy_(
                    s,
                    contract(x, dim=0)
                    + span_indicator[
                        :, torch.arange(n), torch.arange(n) + w
                    ].unsqueeze(-1),
                    w,
                )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ, span_indicator, lens, mbr=mbr
            )
            return {"partition": logZ, "prediction": prediction}

        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_one_new(
        self,
        rules,
        terms,
        lens,
        dropout=None,
        tree=None,
        viterbi=False,
        mbr=False,
    ):
        assert viterbi == False

        # terms = rules['unary']
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        rule = rule.exp()
        X_Y_Z = rule[:, :, NTs, NTs].contiguous()
        X_y_Z = rule[:, :, Ts, NTs].contiguous()
        X_Y_z = rule[:, :, NTs, Ts].contiguous()
        X_y_z = rule[:, :, Ts, Ts].contiguous()

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(
            viterbi or mbr
        )

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ
        @checkpoint
        def Xyz(y, z, rule):
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            x = torch.einsum("bny, bnz, bxyz -> bnx", y, z, rule)
            x = (
                (x + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return x

        @checkpoint
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y, Z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            X = X.logsumexp(2)
            return X

        @checkpoint
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz->bnx", Y, z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return X

        @checkpoint
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz-> bnx", y, Z, rule)
            X = (
                (X + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            return X

        for w in range(2, N):
            n = N - w

            width_idx = terms.new_ones(batch * n).bool()
            ts = [
                s[0].item() + i * n
                for i, b in enumerate(tree)
                for s in b
                if s.diff() == w
            ]
            width_idx[ts] = False
            width_idx = width_idx.reshape(batch, n, 1)

            Y_term = terms[
                :,
                :n,
                :,
            ]
            Z_term = terms[:, w - 1 :, :]
            Y_term = Y_term.masked_fill(width_idx, -1e9)
            Z_term = Z_term.masked_fill(width_idx, -1e9)

            if w == 2:
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[
                        :, torch.arange(n), torch.arange(n) + w
                    ].unsqueeze(-1),
                    w,
                )
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()
            Y = Y.masked_fill(width_idx[..., None], -1e9)
            Z = Z.masked_fill(width_idx[..., None], -1e9)

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[
                    :, torch.arange(n), torch.arange(n) + w
                ].unsqueeze(-1),
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ, span_indicator, lens, mbr=mbr
            )
            return {"partition": logZ, "prediction": prediction}

        else:
            return {"partition": logZ}
