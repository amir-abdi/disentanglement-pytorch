import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.kl import register_kl, kl_divergence
from torch.distributions.utils import _sum_rightmost
from torch.distributions import Independent


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)


def kl_divergence_mu0_var1(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld


def kl_divergence_mu_var1(mu, logvar, target_mu):
    kld = -0.5 * (1 + logvar - (mu - target_mu) ** 2 - logvar.exp()).sum(1).mean()
    return kld


def kl_divergence_var1(logvar):
    kld = -0.5 * (1 + logvar - logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for i, z_j in enumerate(z.split(1, 1)):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    out = torch.cat(perm_z, 1)

    return out


def permute_batch(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm_idx = torch.randperm(B)
    permuted_z = z[perm_idx, :]
    return permuted_z, perm_idx


def entropy(x):
    b = -F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    bsum = b.sum(dim=1)
    return bsum.mean()


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x