import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Flatten3D

class SimpleEncoder64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(256, latent_dim, bias=True)
        )
        # output shape = bs x 256 x 1 x 1

        self.init_layers()

    def forward(self, x):
        return self.main(x)
