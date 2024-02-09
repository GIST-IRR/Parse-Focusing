import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy(p):
    return - torch.sum(p.exp() * p, dim=-1)

def cross_entropy(p, q):
    return - torch.sum(p.exp() * q, dim=-1)

def kl_div(p, q):
    return torch.sum(p.exp() * (p - q), dim=-1)

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_div(p, m) + kl_div(q, m))

def pairwise_cross_entropy(p, q=None, log=False, batch=False):
    if batch:
        b, n, k = p.shape
    else:
        n, k = p.shape

    if q is None:
        q, m = p, n

    if batch:
        return -torch.sum(p.unsqueeze(2).exp() * q.unsqueeze(1), dim=-1)
    else:
        return -torch.sum(p.unsqueeze(1).exp() * q.unsqueeze(0), dim=-1)

def pairwise_kl_divergence(p, q=None, log=False, batch=False):
    if batch:
        b, n, k = p.shape
    else:
        n, k = p.shape
    
    if q is None:
        q, m = p, n
        
    if batch:
        p_q = p.unsqueeze(2) - q.unsqueeze(1) # b, n, m, k
        e_p = p.exp().unsqueeze(2).expand(b, n, m, k)
    else:
        p_q = p.unsqueeze(1) - q.unsqueeze(0) # n, m, k
        e_p = p.exp().unsqueeze(1).expand(n, m, k)

    return torch.sum(e_p * p_q, dim=-1)

def pairwise_js_div(p):
    n, k = p.shape
    m = (p.unsqueeze(1) + p.unsqueeze(0)) / 2 # n, n, k
    #TODO
    return

def pairwise_js_div(p):
    n = p.shape[0]
    res = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            res[i, j] = jensen_shannon_divergence(p[i], p[j])

    return res

def mutual_information(
        p, q, lamb=1.0, eps=sys.float_info.epsilon, normalize=False
    ):
    _, k = p.size()

    # joint probability distribution
    p_i_j = joint_distribution(p, q)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.logsumexp(dim=1)
    p_j = p_i_j.logsumexp(dim=0)

    if normalize:
        h_p_i = - torch.sum(p_i.exp() * p_i)
        h_p_j = - torch.sum(p_j.exp() * p_j)
        h_i_j = torch.minimum(h_p_i, h_p_j)

    p_i = p_i.unsqueeze(1).expand(k, k)
    p_j = p_j.unsqueeze(0).expand(k, k)

    mi = kl_div(p_i_j, p_i + p_j).sum()
    
    if normalize:
        mi = mi / h_i_j
    # mi_no_lamb = kl_div(p_i_j, lamb * (p_i + p_j))
    return mi

def pairwise_mutual_information(p, normalize=False):
    n = p.shape[0]
    mi_mat = torch.zeros(n, n)
    # for i in range(n):
    #     for j in range(n):
    #         mi_mat[i, j] = mutual_information(
    #             p[i].flatten().unsqueeze(0),
    #             p[j].flatten().unsqueeze(0),
    #             normalize=normalize
    #         )
    for i in range(n):
        for j in range(i+1, n):
            mi_mat[i, j] = mutual_information(
                p[i].flatten().unsqueeze(0),
                p[j].flatten().unsqueeze(0),
                normalize=normalize
            )
            mi_mat[j, i] = mi_mat[i, j]
        
def n_pairwise_mutual_information(p, normalize=False):
    n, k, _ = p.shape
    p = p.reshape(n, -1)
    k = k**2

    # joint distribution
    idx = [i for i in range(n) for j in range(i+1, n)]
    idx_t = [j for i in range(n) for j in range(i+1, n)]
    p_i = p[idx].unsqueeze(2)
    p_j = p[idx_t].unsqueeze(1)
    p_i_j = p_i + p_j
    p_i_j = torch.logaddexp(p_i_j, p_i_j.permute(0, 2, 1)) \
        - torch.tensor([2], device=p_i_j.device).log()
    p_i_j = p_i_j - p_i_j.logsumexp(dim=2, keepdim=True).logsumexp(dim=1, keepdim=True)

    m = (n**2 - n) // 2
    p_i = p_i_j.logsumexp(dim=2).unsqueeze(2).expand(m, k, k).reshape(m, -1)
    p_j = p_i_j.logsumexp(dim=1).unsqueeze(1).expand(m, k, k).reshape(m, -1)

    mi = kl_div(p_i_j.reshape(m, -1), p_i + p_j)
    return mi

def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - kl_div(p_i_j.log(), lamb * (p_i * p_j).log()).sum()
    # loss = - p_i_j * (torch.log(p_i_j) \
    #                     - lamb * torch.log(p_j) \
    #                     - lamb * torch.log(p_i))

    # loss = loss.sum()

    loss_no_lamb = - kl_div(p_i_j.log(), (p_i * p_j).log()).sum()
    # loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
    #                             - torch.log(p_j) \
    #                             - torch.log(p_i))

    # loss_no_lamb = loss_no_lamb.sum()

    return loss, loss_no_lamb

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

def joint_distribution(p, q):
    bn, k = p.size()
    assert (q.size(0) == bn and q.size(1) == k)

    p_i_j = p.unsqueeze(2) + q.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.logsumexp(dim=0)
    # symmetrise
    p_i_j = torch.logaddexp(p_i_j, p_i_j.t()) \
        - torch.tensor([2], device=p_i_j.device).log() 
    p_i_j = p_i_j - p_i_j.flatten().logsumexp(dim=0)  # normalise

    return p_i_j

def cosine_similarity(x, y, dim=1, eps=1e-8):
    w12 = torch.sum(x * y, dim)
    w1 = torch.norm(x, 2, dim)
    w2 = torch.norm(y, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).mean()

# def pairwise_cosine_similarity(x, eps=1e-8):
#     x_x = x @ x.T
#     x_norm = torch.diagonal(x_x).sqrt()
#     norms = x_norm.unsqueeze(1) * x_norm.unsqueeze(0)
#     return x_x / norms.clamp(min=eps)

def pairwise_cosine_similarity(x, y=None, eps=1e-8, batch=False):
    if y is None:
        y = x
        if batch:
            x_y = x @ y.permute(0, 2, 1)
        else:
            x_y = x @ y.T
        x_norm = torch.diagonal(x_y).sqrt()
        norms = x_norm.unsqueeze(1) * x_norm.unsqueeze(0)
        return x_y / norms.clamp(min=eps)
    else:
        if batch:
            x_y = x @ y.permute(0, 2, 1)
            x_norm = torch.linalg.norm(x, axis=2)
            y_norm = torch.linalg.norm(y, axis=2)
            norms = x_norm.unsqueeze(2) * y_norm.unsqueeze(1)
        else:
            x_y = x @ y.T
            # x_norm = torch.linalg.norm(x, axis=1)
            # y_norm = torch.linalg.norm(y, axis=1)
            # norms = x_norm.unsqueeze(1) * y_norm.unsqueeze(0)
        # return x_y / norms.clamp(min=eps)
        return x_y