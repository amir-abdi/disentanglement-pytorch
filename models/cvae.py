import logging

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as f

from models.base.base_disentangler import BaseDisentangler
from models.vae import VAE
from architectures import encoders, decoders, others
from common.ops import kl_divergence_mu0_var1, reparametrize
from common.utils import one_hot_embedding


class CVAEModel(nn.Module):
    def __init__(self, encoder, decoder, tiler, num_classes):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tiler = tiler
        self.num_classes = num_classes

    def encode(self, x, y):
        """
        :param x: input data
        :param y: labels with dtype=long, where the number indicates the class of the input (i.e. not one-hot-encoded)
        :return: latent encoding of the input and labels
        """
        # todo: hard coded assuming sinlge label
        y_onehot = one_hot_embedding(y, self.num_classes).squeeze(1)
        y_tiled = self.tiler(y_onehot)
        xy = torch.cat((x, y_tiled), dim=1)
        return self.encoder(xy)

    def decode(self, z, y):
        """

        :param z: latent vector
        :param y: labels with dtype=long, where the number indicates the class of the input (i.e. not one-hot-encoded)
        :return: reconstructed data
        """
        # todo: hard coded assuming sinlge label
        y_onehot = one_hot_embedding(y, self.num_classes).squeeze(1)
        zy = torch.cat((z, y_onehot), dim=1)
        return self.decoder(zy)

    def forward(self, x, y):
        z = self.encode(x, y)
        return self.decode(z)


class CVAE(VAE):
    """
    Auto-Encoding Variational Bayes
    by Kingma and Welling
    https://arxiv.org/pdf/1312.6114.pdf
    """

    def __init__(self, args):
        super().__init__(args)

        # checks
        assert self.num_classes is not None, 'please identify the number of classes for each label separated by comma'

        # hyper-parameters
        # self.w_kld = args.w_kld

        # encoder and decoder
        encoder_name = args.encoder
        decoder_name = args.decoder
        label_tiler_name = args.label_tiler
        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)
        tile_network = getattr(others, label_tiler_name)

        # number of channels
        image_channels = self.num_channels
        label_channels = sum(self.num_classes)
        input_channels = image_channels + label_channels
        decoder_input_channels = self.z_dim + label_channels

        # model and optimizer
        self.model = CVAEModel(encoder(self.z_dim, input_channels, self.image_size),
                               decoder(decoder_input_channels, self.num_channels, self.image_size),
                               tile_network(label_channels, self.image_size),
                               self.num_classes).to(self.device)
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
        labels = kwargs['labels']
        mu, logvar = self.model.encode(images, labels)
        return mu

    def decode(self, **kwargs):
        latent = kwargs['latent']
        labels = kwargs['labels']
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        return self.model.decode(latent, labels)

    def train(self):
        while self.iter < self.max_iter:
            self.net_mode(train=True)
            for x_true1, _, label1, _ in self.data_loader:
                x_true1 = x_true1.to(self.device)
                label1 = label1.to(self.device, dtype=torch.long)

                mu, logvar = self.model.encode(x_true1, label1)
                z = reparametrize(mu, logvar)
                x_recon = torch.sigmoid(self.model.decode(z, label1))

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
        raise NotImplementedError
        self.net_mode(train=False)
        for x_true1, _ in self.data_loader:
            self.iter += 1
            self.pbar.update(1)

            x_true1 = x_true1.to(self.device)
            x_recon = self.model(x_true1)

            self.visualize_recon(x_true1, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true1, None), test=True)
