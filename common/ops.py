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
    """
    This function is borrowed from the implementation by WonKwang Lee
    https://github.com/1Konny/FactorVAE/blob/master/ops.py
    :param z: Input latent vector
    :return: Permuted version of the latent vector
    """
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


def covariance_z_mean(z_mean):
    """Computes the covariance of z_mean.
    Borrowed from https://github.com/google-research/disentanglement_lib/
    Uses cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T.
    Args:
      z_mean: Encoder mean, tensor of size [batch_size, num_latent].
    Returns:
      cov_z_mean: Covariance of encoder mean, tensor of size [num_latent, num_latent].
    """
    expectation_z_mean_z_mean_t = torch.mean(z_mean.unsqueeze(2) * z_mean.unsqueeze(1), dim=0)
    expectation_z_mean = torch.mean(z_mean, dim=0)
    cov_z_mean = expectation_z_mean_z_mean_t - \
                 (expectation_z_mean.unsqueeze(1) * expectation_z_mean.unsqueeze(0))
    return cov_z_mean


def diag_part(tensor):
    """
    return the diagonal elements of batched 2D square matrices
    :param tensor: batched 2D square matrix
    :return: diagonal elements of the matrix
    """
    assert len(tensor.shape) == 2, 'This is implemented for 2D matrices. Input shape is {}'.format(tensor.shape)
    assert tensor.shape[0] == tensor.shape[1], 'This only handles square matrices'
    return tensor[range(len(tensor)), range(len(tensor))]


def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
    """Compute on and off diagonal regularizers for DIP-VAE models.
    Penalize deviations of covariance_matrix from the identity matrix. Uses
    different weights for the deviations of the diagonal and off diagonal entries.
    Borrowed from https://github.com/google-research/disentanglement_lib/

    Args:
      covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
      lambda_od: Weight of penalty for off diagonal elements.
      lambda_d: Weight of penalty for diagonal elements.
    Returns:
      dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
    """
    covariance_matrix_diagonal = diag_part(covariance_matrix)
    # we can set diagonal to zero too... problem with the gradient?
    covariance_matrix_off_diagonal = covariance_matrix - torch.diag(covariance_matrix_diagonal)

    dip_regularizer = lambda_od * torch.sum(covariance_matrix_off_diagonal ** 2) + \
                      lambda_d * torch.sum((covariance_matrix_diagonal - 1) ** 2)

    return dip_regularizer


def total_correlation(z, z_mean, z_logvar):
    """Estimate of total correlation on a batch.
    Borrowed from https://github.com/google-research/disentanglement_lib/
    Args:
      z: [batch_size, num_latents]-tensor with sampled representation.
      z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
      z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
    Returns:
      Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    log_qz_prob = gaussian_log_density(z.unsqueeze(dim=1),
                                       z_mean.unsqueeze(dim=0),
                                       z_logvar.unsqueeze(dim=0))

    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = log_qz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)

    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = log_qz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()

    return (log_qz - log_qz_product).mean()


def gaussian_log_density(samples, mean, log_var):
    """ Estimate the log density of a Gaussian distribution
    Borrowed from https://github.com/google-research/disentanglement_lib/
    :param samples: batched samples of the Gaussian densities with mean=mean and log of variance = log_var
    :param mean: batched means of Gaussian densities
    :param log_var: batches means of log_vars
    :return:
    """
    import math
    pi = torch.tensor(math.pi, requires_grad=False)
    normalization = torch.log(2. * pi)
    inv_sigma = torch.exp(-log_var)
    tmp = samples - mean
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)
