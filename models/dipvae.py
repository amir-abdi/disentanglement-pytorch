import torch


def dipvaei_loss_fn(w_dipvae, lambda_od, lambda_d, **kwargs):
    """
    Variational Inference of Disentangled Latent Concepts from Unlabeled Observations
    by Abhishek Kumar, Prasanna Sattigeri, Avinash Balakrishnan
    https://openreview.net/forum?id=H1kG7GZAW
    :param w_dipvae:
    :param lambda_od:
    :param lambda_d:
    :param kwargs:
    :return:
    """
    mu = kwargs['mu']

    cov_z_mean = covariance_z_mean(mu)
    cov_dip_regularizer = regularize_diag_off_diag_dip(cov_z_mean, lambda_od, lambda_d)
    return cov_dip_regularizer * w_dipvae


def dipvaeii_loss_fn(w_dipvae, lambda_od, lambda_d, **kwargs):
    """
    Variational Inference of Disentangled Latent Concepts from Unlabeled Observations
    by Abhishek Kumar, Prasanna Sattigeri, Avinash Balakrishnan
    https://openreview.net/forum?id=H1kG7GZAW
    :param w_dipvae:
    :param lambda_od:
    :param lambda_d:
    :param kwargs:
    :return:
    """
    mu = kwargs['mu']
    logvar = kwargs['logvar']

    cov_z_mean = covariance_z_mean(mu)
    cov_enc = torch.diag(torch.exp(logvar))
    expectation_cov_enc = torch.mean(cov_enc, dim=0)
    cov_z = expectation_cov_enc + cov_z_mean
    cov_dip_regularizer = regularize_diag_off_diag_dip(cov_z, lambda_od, lambda_d)

    return cov_dip_regularizer * w_dipvae


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


def diag_part(tensor):
    """
    return the diagonal elements of batched 2D square matrices
    :param tensor: batched 2D square matrix
    :return: diagonal elements of the matrix
    """
    assert len(tensor.shape) == 2, 'This is implemented for 2D matrices. Input shape is {}'.format(tensor.shape)
    assert tensor.shape[0] == tensor.shape[1], 'This only handles square matrices'
    return tensor[range(len(tensor)), range(len(tensor))]


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
