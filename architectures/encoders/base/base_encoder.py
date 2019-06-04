import torch.nn as nn


class BaseImageEncoder(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.image_size = image_size

    def forward(self, *input):
        raise NotImplementedError