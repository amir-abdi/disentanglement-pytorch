import torch
from torch import nn
import torch.optim as optim

from models.vae import VAE
from architectures import encoders, decoders, others, discriminators
from common.ops import reparametrize, cross_entropy_multi_label, class_acc_multi_label
from common.utils import one_hot_embedding
from common import constants as c


class IFCVAEModel(nn.Module):
    """
    Adversarial Information Factorization
    by Creswell et al.
    https://arxiv.org/pdf/1711.05175.pdf
    TODO: comment on why the paper might be flawed due to doubly feed forward mechanism
    """
    def __init__(self, z_encoder, label_encoder, decoder, tiler, num_classes):
        super().__init__()

        self.z_encoder = z_encoder
        self.label_encoder = label_encoder
        self.decoder = decoder
        self.tiler = tiler
        self.num_classes = num_classes

        self.total_classes = sum(num_classes)

    def one_hot(self, c):
        if c.size(1) == self.total_classes:
            # c is already one_hot encoded
            return c
        return one_hot_embedding(c, self.num_classes).squeeze(1)

    def encode(self, x, **kwargs):
        """
        :param x: input data
        :return: latent encoding of the input and y
        """
        encode_c = kwargs.get('encode_c', False)
        mu_logvar = self.z_encoder(x)
        if encode_c:
            c_onehot = self.encode_label(x)
            return mu_logvar, c_onehot
        return mu_logvar

    def encode_label(self, x, **kwargs):
        """
        :param x: input data
        :return: label in a one-hot form
        """
        return self.label_encoder(x)

    def encode_z(self, x, **kwargs):
        """
        :param x: input data
        :return: latent vector (z)
        """
        return self.z_encoder(x)

    def decode(self, z, c=None, **kwargs):
        """
        :param z: latent vector
        :param c: labels with dtype=long, where the value indicates the class of the input (i.e. not one-hot-encoded)
        :param c_onehot:  one-hot version of the label
        :return: reconstructed data
        """
        # if c_onehot is None and c is not None:
        #     c_onehot = one_hot_embedding(c, self.num_classes).squeeze(1)

        if c is None:
            # z contains the entire latent space (z + condition)
            assert z.size(1) == self.z_encoder.latent_dim() + self.label_encoder.latent_dim()
            return torch.sigmoid(self.decoder(z))

        c_onehot = self.one_hot(c)
        zy = torch.cat((z, c_onehot), dim=1)
        return torch.sigmoid(self.decoder(zy))

    def forward(self, x, c):
        mu_logvar, y_onehot = self.encode(x, encode_c=True)
        mu, logvar = mu_logvar
        z = reparametrize(mu, logvar)

        return self.decode(z=z, c=y_onehot)


class IFCVAE(VAE):
    """
    Inspired by the rejected ICLR paper "ADVERSARIAL INFORMATION FACTORIZATION"
    by Creswell et al.
    https://arxiv.org/pdf/1711.05175.pdf

    This model excludes the GAN loss from the original paper from the discriminator at the end of the architecture
    which validates the Real/Fakeness of the generated image. The paper referred to this approach as IFCVAE-GAN, thus,
    following the same trend, we call this IFCVAE.
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
        encoder_z = args.encoder[0]
        encoder_label = args.encoder[1]
        decoder = args.decoder[0]
        discriminator_name = args.discriminator[0]
        label_tiler_name = args.label_tiler[0]

        encoder_z = getattr(encoders, encoder_z)
        encoder_l = getattr(encoders, encoder_label)
        decoder = getattr(decoders, decoder)
        tile_network = getattr(others, label_tiler_name)
        discriminator = getattr(discriminators, discriminator_name)

        # number of channels
        image_channels = self.num_channels
        label_channels = self.total_num_classes
        decoder_input_channels = self.z_dim + label_channels

        # model and optimizer
        self.model = IFCVAEModel(z_encoder=encoder_z(self.z_dim, image_channels, self.image_size),
                                 label_encoder=encoder_l(self.total_num_classes, image_channels, self.image_size),
                                 decoder=decoder(decoder_input_channels, self.num_channels, self.image_size),
                                 tiler=tile_network(label_channels, self.image_size),
                                 num_classes=self.num_classes).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # Auxiliary discriminator on z
        self.aux_D = discriminator(self.z_dim, num_classes=self.total_num_classes,
                                   num_layers=self.num_layer_disc,
                                   layer_size=self.size_layer_disc).to(self.device)
        self.optim_aux_D = optim.Adam(self.aux_D.parameters(), lr=self.lr_D, betas=(self.beta1, self.beta2))

        # nets
        self.net_dict.update({
            'G': self.model,
            'aux_D': self.aux_D
        })
        self.optim_dict.update({
            'optim_G': self.optim_G,
            'optim_aux_D': self.optim_aux_D
        })

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if images.dim() == 3:
            images = images.unsqueeze(0)

        mu_logvar = self.model.encode(x=images, encode_c=False)
        mu, _ = mu_logvar
        return mu

    def decode(self, **kwargs):
        latent = kwargs['latent']
        labels = kwargs['labels']

        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        # if z contains both latent encoding and the label encoding, ignore the explicit label (condition)
        if latent.size(1) == self.z_dim + self.total_num_classes:
            return self.model.decode(z=latent)

        return self.model.decode(z=latent, c=labels)

    def train(self):
        while not self.training_complete():
            self.net_mode(train=True)
            for x_true1, label1 in self.data_loader:
                losses = dict()
                accuracies_dict = dict()

                x_true1 = x_true1.to(self.device)
                label1 = label1.to(self.device, dtype=torch.long)
                x_true2, label2 = next(iter(self.data_loader))
                x_true2 = x_true2.to(self.device)
                label2 = label2.to(self.device)

                # train the label encoder bce(y, y_hat)
                y_logits_hat = self.model.encode_label(x=x_true1)
                label_loss = cross_entropy_multi_label(y_logits_hat, label1, self.num_classes) * self.w_le
                accuracies_dict['label_orig'] = class_acc_multi_label(y_logits_hat, label1, self.num_classes)

                # Apply softmax to create soft one_hot encoding to enable condition traversing
                y_onehot_hat = y_logits_hat.clone().detach().softmax(dim=1)

                self.optim_G.zero_grad()  # only zeroing the gradients, the rest should be fine
                label_loss.backward(retain_graph=False)
                self.optim_G.step()
                losses['label'] = label_loss

                # train the main VAE
                losses, params = self.vae_base(losses, x_true1, x_true2, y_onehot_hat, label2)
                x_recon = params['x_recon']
                z = params['z']

                y_onehot_hat_hat = self.model.encode_label(x=x_recon)
                losses['label_hat'] = cross_entropy_multi_label(y_onehot_hat_hat, label1, self.num_classes) * self.w_le
                accuracies_dict['label_recon'] = class_acc_multi_label(y_onehot_hat_hat, label1, self.num_classes)

                aux_y_onehot_hat = self.aux_D(z)
                losses['aux'] = cross_entropy_multi_label(aux_y_onehot_hat, label1, self.num_classes) * self.w_aux

                total_loss = losses[c.TOTAL_VAE] + losses['label_hat'] - losses['aux']

                self.optim_G.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optim_G.step()

                # train auxiliary discriminator
                self.optim_aux_D.zero_grad()
                losses['aux'].backward()
                self.optim_aux_D.step()
                accuracies_dict['auxiliary'] = class_acc_multi_label(aux_y_onehot_hat, label1, self.num_classes)

                # logging and visualization
                self.log_save(input_image=x_true1,
                              recon_image=x_recon,
                              loss=losses,
                              acc=accuracies_dict
                              )
            # end of epoch
        self.pbar.close()

    def test(self):
        self.net_mode(train=False)
        for x_true, label in self.data_loader:
            x_true = x_true.to(self.device)
            label = label.to(self.device)

            x_recon = self.model(x=x_true, c=label)

            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)

            self.iter += 1
            self.pbar.update(1)
