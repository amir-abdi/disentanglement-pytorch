import torch.nn as nn
import torch.nn.functional

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Flatten3D
from common.utils import init_layers


class Encoder_Conv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'
        #assert latent_dim == 64, 'This is the standard for CelebA_64x64'
        self.latent_dim_ = latent_dim
        self.num_channels=num_channels
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 5, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 5, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 5, 2),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            Flatten3D(), ## ADDED MORE LINEAR LAYERS ##
            nn.Linear(1024, latent_dim, bias=True)
        )

        init_layers(self._modules)

    def forward(self, x):
        h = self.main(x)
        return self.main(x)


class EncConv64(Encoder_Conv64):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        print("Created the simple conv64 with channels", num_channels)
        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
#        print("Inside forward in simple_conv64")
 #       print("type", type(x))
  #      print("size", x.size())
        mu_logvar = self.main(x)
        #print("Passed mu_logvar")
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]

        return mu, logvar
