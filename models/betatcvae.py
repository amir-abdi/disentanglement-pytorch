import torch


def betatcvae_loss_fn(w_tc, **kwargs):
    """
    β-TCVAE - Isolating Sources of Disentanglement in Variational Autoencoders
    by Ricky T. Q. Chen, Xuechen Li, Roger Grosse, David Duvenaud
    https://arxiv.org/abs/1802.04942
    :param w_tc:
    :param kwargs:
    :return:
    """
    mu = kwargs['mu']
    logvar = kwargs['logvar']
    z = kwargs['z']

    # todo: double check the (w_tc - 1)
    return total_correlation(z, mu, logvar) * (w_tc - 1)


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
