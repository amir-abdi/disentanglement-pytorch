import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.utils import init_layers
from common.ops import Reshape


class ShallowLinear(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, image_size * image_size * num_channels),
            Reshape([num_channels, image_size, image_size])
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)


class DeepLinear(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Linear(1000, image_size * image_size * num_channels),
            Reshape([num_channels, image_size, image_size])
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)
