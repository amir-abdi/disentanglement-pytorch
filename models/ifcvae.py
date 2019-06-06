# TODO: comment on why the paper might be flawed in terms of the twice feed forward pattern

import logging

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from models.vae import VAE
from architectures import encoders, decoders, others, discriminators
from common.ops import reparametrize, cross_entropy_multi_label, classification_accuracy_multi_label
from common.utils import one_hot_embedding


class IFCVAEModel(nn.Module):
    def __init__(self, z_encoder, label_encoder, decoder, tiler, num_classes):
        super().__init__()

        self.z_encoder = z_encoder
        self.label_encoder = label_encoder
        self.decoder = decoder
        self.tiler = tiler
        self.num_classes = num_classes

    def encode(self, x):
        """
        :param x: input data
        :return: latent encoding of the input and y
        """
        y_onehot = self.label_encoder(x)
        mu_logvar = self.z_encoder(x)
        return mu_logvar, y_onehot

    def encode_label(self, x):
        """
        :param x: input data
        :return: label in a one-hot form
        """
        return self.label_encoder(x)

    def decode(self, z, y=None, y_onehot=None):
        """
        :param z: latent vector
        :param y: labels with dtype=long, where the value indicates the class of the input (i.e. not one-hot-encoded)
        :param y_onehot:  one-hot version of the label
        :return: reconstructed data
        """
        if y_onehot is None and y is not None:
            y_onehot = one_hot_embedding(y, self.num_classes).squeeze(1)
        if y_onehot is None and y is None:
            # z represents the entire latent space
            assert z.size(1) == self.z_encoder.latent_dim() + self.label_encoder.latent_dim()
            return self.decoder(z)

        zy = torch.cat((z, y_onehot), dim=1)
        return self.decoder(zy)

    def forward(self, x, y):
        mu_logvar, y_onehot = self.encode(x)
        mu, logvar = mu_logvar
        z = reparametrize(mu, logvar)

        return self.decode(z, y_onehot=y_onehot)


class IFCVAE(VAE):
    """
    ADVERSARIAL INFORMATION FACTORIZATION
    by Creswell et al.
    https://arxiv.org/pdf/1711.05175.pdf

    Without the GAN loss at the end of the architecture to validate the Real/Fakeness of the generated image. The
    paper referred to this approach as IFCVAE-GAN, thus, following the same trend, we call this IFCVAE.
    """

    def __init__(self, args):
        super().__init__(args)

        # checks
        assert self.num_classes is not None, 'please identify the number of classes for each label separated by comma'

        # hyper-parameters
        self.w_le = args.w_le
        self.w_aux = args.w_aux
        self.num_layer_disc = args.num_layer_disc
        self.size_layer_disc = args.size_layer_disc

        # encoder and decoder
        encoder_name_z = args.encoder[0]
        encoder_name_label = args.encoder[1]
        decoder_name = args.decoder[0]
        discriminator_name = args.discriminator[0]
        label_tiler_name = args.label_tiler[0]

        encoder_z = getattr(encoders, encoder_name_z)
        encoder_l = getattr(encoders, encoder_name_label)
        decoder = getattr(decoders, decoder_name)
        tile_network = getattr(others, label_tiler_name)
        discriminator = getattr(discriminators, discriminator_name)

        # total number of classes
        total_num_classes = sum(self.data_loader.dataset.num_classes(False))
        print('self.num_classes', self.num_classes)
        print('total_num_classes', total_num_classes)

        # number of channels
        image_channels = self.num_channels
        label_channels = total_num_classes
        # input_channels = image_channels + label_channels
        decoder_input_channels = self.z_dim + label_channels

        # model and optimizer
        self.model = IFCVAEModel(z_encoder=encoder_z(self.z_dim, image_channels, self.image_size),
                                 label_encoder=encoder_l(total_num_classes, image_channels, self.image_size),
                                 decoder=decoder(decoder_input_channels, self.num_channels, self.image_size),
                                 tiler=tile_network(label_channels, self.image_size),
                                 num_classes=self.num_classes).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # Auxiliary discriminator on z
        self.aux_D = discriminator(self.z_dim, num_classes=total_num_classes,
                                   num_layers=self.num_layer_disc,
                                   layer_size=self.size_layer_disc).to(self.device)
        self.optim_aux_D = optim.Adam(self.aux_D.parameters(), lr=self.lr_D, betas=(self.beta1, self.beta2))

        # nets
        self.net_dict = {
            'G': self.model,
            'aux_D': self.aux_D
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
            'optim_aux_D': self.optim_aux_D
        }

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)

        mu_logvar, y_onehot = self.model.encode(images)
        mu, _ = mu_logvar
        return torch.cat((mu, y_onehot), dim=1)

    def decode(self, **kwargs):
        latent = kwargs['latent']
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        return self.model.decode(latent)

    def train(self):
        while self.iter < self.max_iter:
            self.net_mode(train=True)
            for x_true, _, label, _ in self.data_loader:
                x_true = x_true.to(self.device)
                label = label.to(self.device, dtype=torch.long)

                mu_logvar, y_onehot_hat = self.model.encode(x_true)
                mu, logvar = mu_logvar
                z = reparametrize(mu, logvar)
                x_recon = torch.sigmoid(self.model.decode(z, y_onehot=y_onehot_hat))

                # train the label encoder bce(y, y_hat) on all the labels
                label_loss = cross_entropy_multi_label(y_onehot_hat, label, self.num_classes) * self.w_le
                self.optim_G.zero_grad()  # only zeroing the gradients, the rest should be fine
                label_loss.backward(retain_graph=True)
                self.optim_G.step()
                accuracy_label_encoder = classification_accuracy_multi_label(y_onehot_hat, label, self.num_classes)

                # train everything else
                y_onehot_hat_hat = self.model.encode_label(x_recon)
                label_hat_loss = cross_entropy_multi_label(y_onehot_hat_hat, label, self.num_classes) * self.w_le
                accuracy_label_recon = classification_accuracy_multi_label(y_onehot_hat_hat, label, self.num_classes)

                # todo: mu or z?
                aux_y_onehot_hat = self.aux_D(z)
                aux_loss = cross_entropy_multi_label(aux_y_onehot_hat, label, self.num_classes)
                aux_loss_weighted = aux_loss * self.w_aux

                recon_loss, kld_loss = self.loss_fn(x_recon=x_recon, x_true=x_true, mu=mu, logvar=logvar)
                loss = recon_loss + kld_loss + label_hat_loss - aux_loss_weighted

                self.optim_G.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_G.step()

                # train auxiliary discriminator
                # todo: calc accuracy
                self.optim_aux_D.zero_grad()
                aux_loss.backward()
                self.optim_aux_D.step()
                accuracy_auxiliary = classification_accuracy_multi_label(aux_y_onehot_hat, label, self.num_classes)

                # logging and visualization
                self.log_save(loss=loss.item(),
                              recon_loss=recon_loss.item(),
                              kld_loss=kld_loss.item(),
                              label_hat_loss=label_hat_loss.item(),
                              label_loss=label_loss.item(),
                              aux_loss=aux_loss.item(),
                              accuracy_auxiliary=accuracy_auxiliary,
                              accuracy_label_encoder=accuracy_label_encoder,
                              accuracy_label_recon=accuracy_label_recon,
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