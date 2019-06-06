import logging

import torch
import torch.nn.functional as f
import torch.optim as optim

from models.vae import VAE as VAE
from common.ops import reparametrize, permute_dims
from architectures import discriminators


class FactorVAE(VAE):
    """
    Disentangling by Factorising
    by Kim and Mnih
    https://arxiv.org/pdf/1802.05983.pdf
    """

    def __init__(self, args):
        super().__init__(args)

        # hyper-parameters
        self.num_layer_disc = args.num_layer_disc
        self.size_layer_disc = args.size_layer_disc
        self.w_tc = args.w_tc

        # Permute discriminator network
        discriminator_name = args.discriminator[0]
        discriminator = getattr(discriminators, discriminator_name)
        self.PermD = discriminator(self.z_dim, num_classes=2,
                                   num_layers=self.num_layer_disc,
                                   layer_size=self.size_layer_disc).to(self.device)
        self.optim_PermD = optim.Adam(self.PermD.parameters(), lr=self.lr_D,
                                      betas=(self.beta1, self.beta2))

        self.net_dict.update({'PermD': self.PermD})
        self.optim_dict = {'optim_PermD': self.optim_PermD}

    def loss_fn(self, **kwargs):
        x_recon = kwargs['x_recon']
        x_true = kwargs['x_true']
        mu = kwargs['mu']
        logvar = kwargs['logvar']
        dz_true = kwargs['dz_true']

        recon_loss = f.binary_cross_entropy(x_recon, x_true, reduction='mean') * self.w_recon
        kld_loss = self._kld_loss_fn(mu, logvar)
        vae_tc_loss = (dz_true[:, 0] - dz_true[:, 1]).mean() * self.w_tc

        return recon_loss, kld_loss, vae_tc_loss

    def train(self):
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device, requires_grad=False)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device, requires_grad=False)

        while self.iter < self.max_iter:
            self.net_mode(train=True)
            for x_true1, x_true2 in self.data_loader:
                x_true1 = x_true1.to(self.device)
                x_true2 = x_true2.to(self.device)

                mu, logvar = self.model.encode(x_true1)
                z = reparametrize(mu, logvar)
                x_recon = torch.sigmoid(self.model.decode(z))
                dz_true = self.PermD(z)

                recon_loss, kld_loss, tc_loss = self.loss_fn(x_recon=x_recon, x_true=x_true1, mu=mu,
                                                             logvar=logvar, dz_true=dz_true)
                loss = recon_loss + kld_loss + tc_loss

                self.optim_G.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_G.step()

                # --------- Training the discriminator ----------
                mu2, logvar2 = self.model.encode(x_true2)
                z2 = reparametrize(mu2, logvar2)

                z2_perm = permute_dims(z2).detach()
                dz2_perm = self.PermD(z2_perm)

                tc_loss_discriminator = (f.cross_entropy(dz_true, zeros) + f.cross_entropy(dz2_perm, ones)) * 0.5

                self.optim_PermD.zero_grad()
                tc_loss_discriminator.backward()
                self.optim_PermD.step()

                # --------- Logging and visualization ----------
                self.log_save(loss=loss.item(),
                              recon_loss=recon_loss.item(),
                              kld_loss=kld_loss.item(),
                              vae_tc_loss=tc_loss.item(),
                              tc_loss=tc_loss_discriminator.item(),
                              input_image=x_true1,
                              recon_image=x_recon,
                              )
                self.iter += 1
                self.pbar.update(1)

        logging.info("-------Training Finished----------")
        self.pbar.close()
