import torch
import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
#from common.ops import Unsqueeze3D
from common.utils import init_layers

class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x.view(-1, 1024, 8, 8)

class DeConv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'
        assert latent_dim == 64, 'This is the standard for CelebA_64x64'
        
        self.main =  self.dec = nn.Sequential(
            nn.Linear(latent_dim, 1024*latent_dim),
            nn.ReLU(),
            Unsqueeze3D(),
            nn.ConvTranspose2d(1024, 512,  5, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 5, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 1, 12),
        )
        # output shape = bs x 3 x 64 x 64

        init_layers(self._modules)

    def forward(self, x):
        h  = self.main(x)
        assert h.size(2) == 64, 'Wrong processing'+str(h.size())
        return self.main(x)
