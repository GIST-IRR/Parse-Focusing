import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_scatter import scatter_max
from parser.pcfgs.fn import (
    stripe,
    diagonal_copy_depth,
    diagonal_copy_,
    diagonal_depth,
    diagonal,
    checkpoint,
)


class PCFG_base(nn.Module):

    def _inside(self):
        raise NotImplementedError

    def _inside_topk(self):
        raise NotImplementedError

    def forward(self, rules, terms, lens, topk=None, **kwargs):
        if topk:
            return self._inside_topk(rules, terms, lens, topk=topk, **kwargs)
        elif "tree" in kwargs.keys():
            return self._inside_one(rules, terms, lens, **kwargs)
        else:
            if isinstance(kwargs.get("w2T", None), torch.Tensor):
                return self._inside_weighted(rules, terms, lens, **kwargs)
            else:
                return self._inside(rules, terms, lens, **kwargs)

        return result

    def _get_prediction(
        self,
        logZ,
        span_indicator,
        lens,
        tag_indicator=None,
        mbr=False,
        depth=False,
        label=False,
    ):
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
            tag_marginals = (
                tag_indicator.grad.detach()
                if tag_indicator is not None
                else None
            )
            if mbr:
                if depth:
                    return self._cky_zero_order_depth(marginals.detach(), lens)
                else:
                    # MBR decoding show the difference of perfermance between with and without tagger
                    # Before find the ways to apply tagger without loss of performance, use without tagger
                    if marginals.dim() == 4:
                        if label and tag_marginals is not None:
                            return self._cky_zero_order_label(
                                marginals.detach(),
                                lens,
                                tag_marginals=tag_marginals,
                            )
                        else:
                            return self._cky_zero_order(
                                marginals.detach().sum(-1), lens
                            )
                    elif marginals.dim() == 3:
                        return self._cky_zero_order(marginals.detach(), lens)
                    # return self._cky_zero_order_tag(
                    #     marginals.detach(), tag_marginals.detach(), lens
                    # )
            else:
                viterbi_spans = marginals.nonzero().tolist()
                for span in viterbi_spans:
                    if depth:
                        prediction[span[0]][span[3]].append((span[1], span[2]))
                    else:
                        # prediction[span[0]].append((span[1], span[2]))
                        prediction[span[0]].append((span[1], span[2], span[3]))
                if tag_marginals is not None:
                    viterbi_tags = tag_marginals.nonzero().tolist()
                    for tag in viterbi_tags:
                        prediction[tag[0]].append((tag[1], tag[1] + 1, tag[2]))

        return prediction

    @torch.no_grad()
    def _cky_zero_order(self, marginals, lens):
        N = marginals.shape[-1]
        # s tensor save diagonal elements in marginals tensor first.
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        # p tensor save what?
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
            x = X + diagonal(marginals, w)  #
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
    def _cky_zero_order_label(self, marginals, lens, tag_marginals=None):
        assert marginals.dim() == 4
        l, _, N, NT = marginals.shape
        # s tensor save diagonal elements in marginals tensor first.
        s = marginals.new_full(marginals.shape, -1e9)
        # p tensor save splitting point
        p = marginals.new_zeros(*marginals.shape[:-1]).long()
        # t tensor save label
        t = marginals.new_zeros(*marginals.shape[:-1]).long()
        # Terminal tag marginals
        tt = tag_marginals.argmax(2)

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
            # Choose maximum splitting point
            split = (Y.sum(-1) + Z.sum(-1)).argmax(2)
            label_idx = split[..., None, None].expand(l, n, 1, NT)
            # Choose maximum marginals
            X = (Y + Z).gather(2, label_idx).squeeze(2)
            x = X + diagonal(marginals, w)
            tag = x.max(-1)[1]

            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)
            diagonal_copy_(t, tag, w)

        def backtrack_label(p, i, j, tag, ttag):
            if j - i == 1:
                return [(i, j, ttag[i])]
            split = p[i][j]
            label = tag[i][j]
            ltree = backtrack_label(p, i, split, tag, ttag)
            rtree = backtrack_label(p, split, j, tag, ttag)
            return [(i, j, label)] + ltree + rtree

        p = p.tolist()
        lens = lens.tolist()
        t = t.tolist()
        tt = tt.tolist()
        spans = [
            backtrack_label(p[i], 0, length, t[i], tt[i])
            for i, length in enumerate(lens)
        ]
        return spans

    @torch.no_grad()
    def _cky_zero_order_tag(self, marginals, tag_marginals, lens):
        b, N = marginals.shape[:2]
        NT = marginals.shape[-1]
        T = tag_marginals.shape[-1]

        s = marginals.new_full((*marginals.shape[:-1], NT + T), -1e9)
        p = marginals.new_zeros(*marginals.shape[:-1]).long()
        l = marginals.new_zeros(*marginals.shape[:-1]).long()
        r = marginals.new_zeros(*marginals.shape[:-1]).long()

        x = torch.cat([diagonal(marginals, 1), tag_marginals], dim=-1)
        diagonal_copy_(s, x, 1)

        def calculate_index(Y_Z, index):
            b, n, *_ = Y_Z.shape
            indices = index.new_zeros(3, b, n)
            stride = Y_Z.stride()[2:]
            for i in range(len(stride)):
                indices[i] = torch.div(index, stride[i], rounding_mode="floor")
                index = index % stride[i]
            return indices[0], indices[1], indices[2]

        for w in range(2, N):
            n = N - w
            starts = p.new_tensor(range(n))
            if w == 2:
                Y = stripe(s, n, w - 1, (0, 1)).unsqueeze(4)
                Z = stripe(s, n, w - 1, (1, w), 0).unsqueeze(3)
            else:
                Y = stripe(s, n, w - 1, (0, 1)).unsqueeze(4)
                Z = stripe(s, n, w - 1, (1, w), 0).unsqueeze(3)

            Y_Z = Y + Z
            X, index = Y_Z.reshape(b, n, -1).max(-1)
            split, left, right = calculate_index(Y_Z, index)
            X = X.unsqueeze(-1)

            x = torch.cat(
                [X + diagonal(marginals, w), X.new_zeros(*X.shape[:-1], T)],
                dim=-1,
            )
            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)
            diagonal_copy_(l, left, w)
            diagonal_copy_(r, right, w)

        X, root_tag = s[torch.arange(b), 0, lens].max(-1)

        def backtrack_tag(p, i, j, tag, l, r):
            if j == i + 1:
                return [(i, j, tag - NT)]
            split = p[i][j]
            left = l[i][j]
            right = r[i][j]
            ltree = backtrack_tag(p, i, split, left, l, r)
            rtree = backtrack_tag(p, split, j, right, l, r)
            return [(i, j, tag)] + ltree + rtree

        p = p.tolist()
        l = l.tolist()
        r = r.tolist()
        root_tag = root_tag.tolist()
        lens = lens.tolist()
        spans_tags = [
            backtrack_tag(p[i], 0, length, root_tag[i], l[i], r[i])
            for i, length in enumerate(lens)
        ]
        return spans_tags

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
        """
        :param attach: The marginal probabilities.
        :param lens: sentences lens
        :return: predicted_arcs
        """
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
                [
                    torch.zeros(b, N, N, device=self.device).fill_(self.huge)
                    for _ in range(2)
                ]
                for _ in range(2)
            ]
            for _ in range(2)
        ]
        alpha[A][C][L][:, :, 0] = 0
        alpha[B][C][L][:, :, -1] = 0
        alpha[A][C][R][:, :, 0] = 0
        alpha[B][C][R][:, :, -1] = 0
        semiring_plus = self.get_plus_semiring(viterbi=True)
        # single root.
        start_idx = 1
        for k in range(1, N - start_idx):
            f = torch.arange(start_idx, N - k), torch.arange(k + start_idx, N)
            ACL = alpha[A][C][L][:, start_idx : N - k, :k]
            ACR = alpha[A][C][R][:, start_idx : N - k, :k]
            BCL = alpha[B][C][L][:, start_idx + k :, N - k :]
            BCR = alpha[B][C][R][:, start_idx + k :, N - k :]
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_l = x + attach[:, f[1], f[0]]
            alpha[A][I][L][:, start_idx : N - k, k] = arcs_l
            alpha[B][I][L][:, k + start_idx : N, N - k - 1] = arcs_l
            x = semiring_plus(ACR + BCL, dim=2)
            arcs_r = x + attach[:, f[0], f[1]]
            alpha[A][I][R][:, start_idx : N - k, k] = arcs_r
            alpha[B][I][R][:, k + start_idx : N, N - k - 1] = arcs_r
            AIR = alpha[A][I][R][:, start_idx : N - k, 1 : k + 1]
            BIL = alpha[B][I][L][:, k + start_idx :, N - k - 1 : N - 1]
            new = semiring_plus(ACL + BIL, dim=2)
            alpha[A][C][L][:, start_idx : N - k, k] = new
            alpha[B][C][L][:, k + start_idx : N, N - k - 1] = new
            new = semiring_plus(AIR + BCR, dim=2)
            alpha[A][C][R][:, start_idx : N - k, k] = new
            alpha[B][C][R][:, start_idx + k : N, N - k - 1] = new
        # dealing with the root.
        root_incomplete_span = alpha[A][C][L][:, 1, : N - 1] + attach[:, 0, 1:]
        for k in range(1, N):
            AIR = root_incomplete_span[:, :k]
            BCR = alpha[B][C][R][:, k, N - k :]
            alpha[A][C][R][:, 0, k] = semiring_plus(AIR + BCR, dim=1)
        logZ = torch.gather(alpha[A][C][R][:, 0, :], -1, lens.unsqueeze(-1))
        arc = torch.autograd.grad(logZ.sum(), attach)[0].nonzero().tolist()
        predicted_arc = [[] for _ in range(logZ.shape[0])]
        for a in arc:
            predicted_arc[a[0]].append((a[1] - 1, a[2] - 1))
        return predicted_arc
