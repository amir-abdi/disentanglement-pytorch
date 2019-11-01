import torch
import torch.nn.functional as F
import torch.optim as optim

from common.ops import reparametrize
from architectures import discriminators


def factorvae_init(discriminator_name, z_dim, num_layer_disc, size_layer_disc, lr_D, beta1, beta2):
    assert discriminator_name is not None, 'FactorVAE needs a discriminator to detect permuted Zs'
    discriminator = getattr(discriminators, discriminator_name)

    PermD = discriminator(z_dim, num_classes=2, num_layers=num_layer_disc,
                          layer_size=size_layer_disc)
    optim_PermD = optim.Adam(PermD.parameters(), lr=lr_D, betas=(beta1, beta2))
    return PermD, optim_PermD


def factorvae_loss_fn(w_tc, model, PermD, optim_PermD, ones, zeros, **kwargs):
    """
    Disentangling by Factorising
    by Kim and Mnih
    https://arxiv.org/pdf/1802.05983.pdf
    :param w_tc:
    :param model:
    :param PermD:
    :param optim_PermD:
    :param ones:
    :param zeros:
    :param kwargs:
    :return:
    """
    x_true2 = kwargs['x_true2']
    label2 = kwargs['label2']
    z = kwargs['z']

    factorvae_dz_true = PermD(z)
    vae_tc_loss = (factorvae_dz_true[:, 0] - factorvae_dz_true[:, 1]).mean() * w_tc

    # Train discriminator of FactorVAE
    mu2, logvar2 = model.encode(x=x_true2, c=label2)
    z2 = reparametrize(mu2, logvar2)
    z2_perm = permute_dims(z2).detach()
    dz2_perm = PermD(z2_perm)
    discriminator_tc_loss = (F.cross_entropy(factorvae_dz_true, zeros) +
                             F.cross_entropy(dz2_perm, ones)) * 0.5
    optim_PermD.zero_grad()
    discriminator_tc_loss.backward(retain_graph=True)
    optim_PermD.step()

    return vae_tc_loss, discriminator_tc_loss


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
