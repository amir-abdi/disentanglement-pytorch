import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.utils import init_layers
from common.ops import Flatten3D


class ShallowGaussianLinear(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)

        self.main = nn.Sequential(
            Flatten3D(),
            nn.Linear(image_size * image_size * num_channels, 400),
            nn.ReLU())

        self.head_mu = nn.Linear(400, latent_dim)
        self.head_logvar = nn.Linear(400, latent_dim)

        init_layers(self._modules)

    def forward(self, x):
        h = self.main(x)
        return self.head_mu(h), self.head_logvar(h)


class DeepGaussianLinear(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)

        self.main = nn.Sequential(
            Flatten3D(),
            nn.Linear(image_size * image_size * num_channels, 1000),
            nn.ReLU(),
            nn.Linear(1000, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
        )

        self.head_mu = nn.Linear(100, latent_dim)
        self.head_logvar = nn.Linear(100, latent_dim)

        init_layers(self._modules)

    def forward(self, x):
        h = self.main(x)
        return self.head_mu(h), self.head_logvar(h)
