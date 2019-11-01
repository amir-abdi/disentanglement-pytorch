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


class Reshape(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        s = [x.size(0)]
        s += self.size
        x = x.view(s)
        return x


class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)


def cross_entropy_multi_label(pred, target, num_classes):
    num_labels = len(num_classes)
    loss = 0
    start_idx = 0
    for i in range(num_labels):
        end_idx = start_idx + num_classes[i]
        sub_target = target[:, i]
        sub_pred = pred[:, start_idx:end_idx]

        loss += F.cross_entropy(sub_pred, sub_target)
        start_idx = end_idx
    return loss


def classification_accuracy(pred, target):
    batch_size = pred.size(0)
    _, pred_class = torch.max(pred, 1)
    return (pred_class == target).sum().item() / batch_size


def class_acc_multi_label(pred, target, num_classes):
    num_labels = len(num_classes)
    batch_size = pred.size(0)

    acc = 0
    start_idx = 0
    for i in range(num_labels):
        end_idx = start_idx + num_classes[i]
        sub_target = target[:, i]
        sub_pred = pred[:, start_idx:end_idx]

        _, sub_pred_class = torch.max(sub_pred, 1)
        acc += (sub_pred_class == sub_target).sum().item() / batch_size
        start_idx = end_idx
    return acc / num_labels
