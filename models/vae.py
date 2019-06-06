import logging

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as f

from models.base.base_disentangler import BaseDisentangler
from architectures import encoders, decoders
from common.ops import kl_divergence_mu0_var1, reparametrize


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        return self.decode(z)


class VAE(BaseDisentangler):
    """
    Auto-Encoding Variational Bayes
    by Kingma and Welling
    https://arxiv.org/pdf/1312.6114.pdf
    """

    def __init__(self, args):
        super().__init__(args)

        # hyper-parameters
        self.w_kld = args.w_kld
        self.max_c = torch.tensor(args.max_c, dtype=torch.float)
        self.iterations_c = torch.tensor(args.iterations_c, dtype=torch.float)

        # As a joke and sanity check!
        assert self.w_kld == 1.0 or self.alg != 'VAE', 'in vanilla VAE, w_kld should be 1.0. ' \
                                                       'Please use BetaVAE if intended otherwise.'

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]
        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # model and optimizer
        self.model = VAEModel(encoder(self.z_dim, self.num_channels, self.image_size),
                              decoder(self.z_dim, self.num_channels, self.image_size)).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # nets
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu, logvar = self.model.encode(images)
        return mu

    def encode_stochastic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu, logvar = self.model.encode(images)
        return reparametrize(mu, logvar)

    def _kld_loss_fn(self, mu, logvar):
        if self.vae_loss == 'Basic':
            kld_loss = kl_divergence_mu0_var1(mu, logvar) * self.w_kld
        elif self.vae_loss == 'AnnealedCapacity':
            c = torch.min(self.max_c,
                          self.max_c * torch.tensor(self.iter) / self.iterations_c)
            kld_loss = (kl_divergence_mu0_var1(mu, logvar) - c).abs() * self.w_kld
        else:
            raise NotImplementedError

        return kld_loss

    def loss_fn(self, **kwargs):
        x_recon = kwargs['x_recon']
        x_true = kwargs['x_true']
        mu = kwargs['mu']
        logvar = kwargs['logvar']

        recon_loss = f.binary_cross_entropy(x_recon, x_true, reduction='mean') * self.w_recon
        kld_loss = self._kld_loss_fn(mu, logvar)

        return recon_loss, kld_loss

    def train(self):
        while self.iter < self.max_iter:
            self.net_mode(train=True)
            for x_true1, _ in self.data_loader:
                x_true1 = x_true1.to(self.device)

                mu, logvar = self.model.encode(x_true1)
                z = reparametrize(mu, logvar)
                x_recon = torch.sigmoid(self.model.decode(z))

                recon_loss, kld_loss = self.loss_fn(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar)
                loss = recon_loss + kld_loss

                self.optim_G.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_G.step()

                self.log_save(loss=loss.item(),
                              recon_loss=recon_loss.item(),
                              kld_loss=kld_loss.item(),
                              input_image=x_true1,
                              recon_image=x_recon,
                              )
                self.iter += 1
                self.pbar.update(1)

        logging.info("-------Training Finished----------")
        self.pbar.close()

    def test(self):
        self.net_mode(train=False)
        for x_true1, _ in self.data_loader:
            self.iter += 1
            self.pbar.update(1)

            x_true1 = x_true1.to(self.device)
            x_recon = self.model(x_true1)

            self.visualize_recon(x_true1, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true1, None), test=True)


class BetaVAE(VAE):
    """
    β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
    by Higgins et al.
    https://openreview.net/pdf?id=Sy2fzU9gl

    Understanding disentangling in β-VAE
    by Burgess et al.
    https://arxiv.org/pdf/1804.03599.pdf
    """

    def __init__(self, args):
        super().__init__(args)
