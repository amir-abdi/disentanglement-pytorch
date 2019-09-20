import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from models.base.base_disentangler import BaseDisentangler
from architectures import encoders, decoders, discriminators
import common
from common.ops import kl_divergence_mu0_var1, reparametrize, permute_dims
from common import constants as c


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def decode(self, z, **kwargs):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x, **kwargs):
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

        # FactorVAE
        if c.FACTORVAE in self.vae_type:
            self.PermD, self.optim_PermD = self.factorvae_init(args)
            self.net_dict.update({'PermD': self.PermD})
            self.optim_dict.update({'optim_PermD': self.optim_PermD})

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu, logvar = self.model.encode(x=images)
        return mu

    def encode_stochastic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        mu, logvar = self.model.encode(x=images)
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

    def loss_fn(self, input_losses, **kwargs):
        x_recon = kwargs['x_recon']
        x_true = kwargs['x_true']
        mu = kwargs['mu']
        logvar = kwargs['logvar']

        bs = self.batch_size
        output_losses = dict()
        output_losses[c.TOTAL_VAE] = input_losses.get(c.TOTAL_VAE, 0)

        output_losses[c.RECON] = F.binary_cross_entropy(x_recon, x_true, reduction='sum') / bs * self.w_recon
        output_losses[c.TOTAL_VAE] += output_losses[c.RECON]

        output_losses['kld'] = self._kld_loss_fn(mu, logvar)
        output_losses[c.TOTAL_VAE] += output_losses['kld']

        if c.FACTORVAE in self.vae_type:
            # todo: rename tc values to more descriptive accronyms rather than anal vs empirical
            output_losses['vae_tc'], output_losses['discriminator_tc'] = self._factorvae_loss_fn(**kwargs)
            output_losses[c.TOTAL_VAE] += output_losses['vae_tc']

        if c.DIPVAE in self.vae_type:
            output_losses['vae_dip'] = self._dipvae_loss_fn(**kwargs)
            output_losses[c.TOTAL_VAE] += output_losses['vae_dip']

        if c.BetaTCVAE in self.vae_type:
            output_losses['vae_tc_analytical'] = self._betatcvae_loss_fn(**kwargs)
            output_losses[c.TOTAL_VAE] += output_losses['vae_tc_analytical']

        if c.INFOVAE in self.vae_type:
            output_losses['vae_mmd'] = self._infovae_loss_fn(**kwargs)
            output_losses[c.TOTAL_VAE] += output_losses['vae_mmd']

        return output_losses

    def vae_base(self, losses, x_true1, x_true2, label1, label2):
        mu, logvar = self.model.encode(x=x_true1, c=label1)
        z = reparametrize(mu, logvar)
        x_recon = self.model.decode(z=z, c=label1)
        loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z,
                            x_true2=x_true2, label2=label2)

        losses.update(self.loss_fn(losses, **loss_fn_args))
        return losses, {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar}

    def train(self):
        while not self.training_complete():
            self.net_mode(train=True)
            vae_loss_epoch = 0
            for x_true1, label1 in self.data_loader:
                losses = dict()

                x_true1 = x_true1.to(self.device)
                label1 = label1.to(self.device)
                x_true2, label2 = next(iter(self.data_loader))
                x_true2 = x_true2.to(self.device)
                label2 = label2.to(self.device)

                # if the last batch did not have enough samples, skip it
                if x_true1.shape != x_true2.shape:
                    continue

                losses, params = self.vae_base(losses, x_true1, x_true2, label1, label2)

                self.optim_G.zero_grad()
                losses[c.TOTAL_VAE].backward(retain_graph=False)
                self.optim_G.step()

                self.log_save(input_image=x_true1,
                              recon_image=params['x_recon'],
                              loss=losses)
                vae_loss_epoch += losses[c.TOTAL_VAE]

            # end of epoch
            self.lr_scheduler_step(validation_loss=vae_loss_epoch / self.num_batches)
        self.pbar.close()

    def factorvae_init(self, args):
        """
          Disentangling by Factorising
          by Kim and Mnih
          https://arxiv.org/pdf/1802.05983.pdf
          """
        assert args.discriminator is not None, 'FactorVAE needs a discriminator to detect permuted Zs'
        discriminator_name = args.discriminator[0]
        discriminator = getattr(discriminators, discriminator_name)

        PermD = discriminator(self.z_dim, num_classes=2, num_layers=self.num_layer_disc,
                              layer_size=self.size_layer_disc).to(self.device)
        optim_PermD = optim.Adam(PermD.parameters(), lr=self.lr_D, betas=(self.beta1, self.beta2))
        return PermD, optim_PermD

    def _factorvae_loss_fn(self, **kwargs):
        # todo: add documentation and paper
        x_true2 = kwargs['x_true2']
        label2 = kwargs['label2']
        z = kwargs['z']

        factorvae_dz_true = self.PermD(z)
        vae_tc_loss = (factorvae_dz_true[:, 0] - factorvae_dz_true[:, 1]).mean() * self.w_tc_empirical

        # Train discriminator of FactorVAE
        mu2, logvar2 = self.model.encode(x=x_true2, c=label2)
        z2 = reparametrize(mu2, logvar2)
        z2_perm = permute_dims(z2).detach()
        dz2_perm = self.PermD(z2_perm)
        discriminator_tc_loss = (F.cross_entropy(factorvae_dz_true, self.zeros) +
                                 F.cross_entropy(dz2_perm, self.ones)) * 0.5
        self.optim_PermD.zero_grad()
        discriminator_tc_loss.backward(retain_graph=True)
        self.optim_PermD.step()

        return vae_tc_loss, discriminator_tc_loss

    def _dipvae_loss_fn(self, **kwargs):
        # todo: add documentation and paper
        mu = kwargs['mu']
        logvar = kwargs['logvar']

        from common.ops import covariance_z_mean, regularize_diag_off_diag_dip
        cov_z_mean = covariance_z_mean(mu)

        # todo: get rid of dip_type argument
        if self.dip_type == "i":
            cov_dip_regularizer = regularize_diag_off_diag_dip(cov_z_mean, self.lambda_od, self.lambda_d)
        elif self.dip_type == "ii":
            cov_enc = torch.diag(torch.exp(logvar))
            expectation_cov_enc = torch.mean(cov_enc, dim=0)
            cov_z = expectation_cov_enc + cov_z_mean
            cov_dip_regularizer = regularize_diag_off_diag_dip(cov_z, self.lambda_od, self.lambda_d)
        else:
            raise NotImplementedError("DIP variant not supported.")

        return cov_dip_regularizer * self.w_dipvae

    def _betatcvae_loss_fn(self, **kwargs):
        # todo: add documentation and paper
        mu = kwargs['mu']
        logvar = kwargs['logvar']
        z = kwargs['z']

        # Instead of substracting w_tc_analytical by 1, we just used w_tc_analytical to keep things consistent
        return common.ops.total_correlation(z, mu, logvar) * self.w_tc_analytical

    def _infovae_loss_fn(self, **kwargs):
        # todo: add documentation and paper
        from common.ops import compute_mmd
        z = kwargs['z']
        z_true = torch.randn(1000, self.z_dim).to(self.device)
        return compute_mmd(z_true, z) * self.w_infovae

    def test(self):
        self.net_mode(train=False)
        for x_true, label in self.data_loader:
            x_true = x_true.to(self.device)
            label = label.to(self.device, dtype=torch.long)

            x_recon = self.model(x=x_true, c=label)

            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)

            self.iter += 1
            self.pbar.update(1)


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
