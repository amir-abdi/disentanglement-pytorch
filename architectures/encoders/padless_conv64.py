import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Flatten3D
from common.utils import init_layers


class PadlessEncoder64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, 2, 0),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 0),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 0),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 0),
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(256, latent_dim, bias=True)
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)


class PadlessGaussianConv64(PadlessEncoder64):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar
