import logging

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as f

from models.base.base_disentangler import BaseDisentangler
from architectures import encoders, decoders


class AEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class AE(BaseDisentangler):
    def __init__(self, args):
        super().__init__(args)

        # encoder and decoder
        self.encoder_name = args.encoder[0]
        self.decoder_name = args.decoder[0]
        encoder = getattr(encoders, self.encoder_name)
        decoder = getattr(decoders, self.decoder_name)

        # model and optimizer
        self.model = AEModel(encoder(self.z_dim, self.num_channels, self.image_size),
                             decoder(self.z_dim, self.num_channels, self.image_size)).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # nets
        self.nets = [self.model]
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }

    def loss_fn(self, **kwargs):
        x_recon = kwargs['x_recon']
        x_true = kwargs['x_true']

        recon_loss = f.binary_cross_entropy(x_recon, x_true, reduction='mean') * self.w_recon

        return recon_loss

    def train(self):
        while self.iter < self.max_iter:
            self.net_mode(train=True)
            for x_true1, _ in self.data_loader:
                x_true1 = x_true1.to(self.device)

                x_recon = torch.sigmoid(self.model(x_true1))

                recon_loss = self.loss_fn(x_recon=x_recon, x_true=x_true1)
                loss = recon_loss

                self.optim_G.zero_grad()
                loss.backward(retain_graph=True)
                self.optim_G.step()

                self.log_save(loss=loss.item(),
                              recon_loss=recon_loss.item(),
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
