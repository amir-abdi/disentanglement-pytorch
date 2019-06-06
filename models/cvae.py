import logging

import torch
from torch import nn
import torch.optim as optim

from models.vae import VAE
from architectures import encoders, decoders, others
from common.ops import reparametrize
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
        y_onehot = one_hot_embedding(y, self.num_classes).squeeze(1)
        zy = torch.cat((z, y_onehot), dim=1)
        return self.decoder(zy)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = reparametrize(mu, logvar)
        return self.decode(z, y)


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

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]
        label_tiler_name = args.label_tiler[0]

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
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
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
            for x_true, _, label, _ in self.data_loader:
                x_true = x_true.to(self.device)
                label = label.to(self.device, dtype=torch.long)

                mu, logvar = self.model.encode(x_true, label)
                z = reparametrize(mu, logvar)
                x_recon = torch.sigmoid(self.model.decode(z, label))

                recon_loss, kld_loss = self.loss_fn(x_recon=x_recon, x_true=x_true, mu=mu, logvar=logvar)
                loss = recon_loss + kld_loss

                self.optim_G.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_G.step()

                self.log_save(loss=loss.item(),
                              recon_loss=recon_loss.item(),
                              kld_loss=kld_loss.item(),
                              input_image=x_true,
                              recon_image=x_recon,
                              )
                self.iter += 1
                self.pbar.update(1)

        logging.info("-------Training Finished----------")
        self.pbar.close()

    def test(self):
        self.net_mode(train=False)
        for x_true, _, label, _ in self.data_loader:
            x_true = x_true.to(self.device)
            label = label.to(self.device, dtype=torch.long)

            x_recon = self.model(x_true, label)

            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)

            self.iter += 1
            self.pbar.update(1)
