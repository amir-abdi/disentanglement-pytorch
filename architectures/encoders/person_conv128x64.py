import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Flatten3D
from common.utils import init_layers


class PersonGaussianConv128x64(BaseImageEncoder):
    """
    Convolutional VAE encoder designed specifically for
    pedestrian images with shape 128 x 64.

    Input:
        [batch_size, 3, 128, 64]

    Output:
        mu     -> [batch_size, latent_dim]
        logvar -> [batch_size, latent_dim]
    """

    def __init__(
        self,
        latent_dim,
        num_channels=3,
        image_size=None
    ):
        # image_size is kept for compatibility with the
        # repository's existing architecture interface.
        super().__init__(
            latent_dim,
            num_channels,
            image_size
        )

        self._latent_dim = latent_dim

        self.main = nn.Sequential(

            # ------------------------------------------------
            # Input:
            # [B, 3, 128, 64]
            # ------------------------------------------------

            nn.Conv2d(
                num_channels,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # [B, 32, 64, 32]

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # [B, 64, 32, 16]

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # [B, 128, 16, 8]

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # [B, 256, 8, 4]

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # [B, 512, 4, 2]

            Flatten3D(),

            # 512 * 4 * 2 = 4096

            nn.Linear(
                4096,
                latent_dim * 2
            )
        )

        init_layers(
            self._modules
        )

    def forward(self, x):

        mu_logvar = self.main(x)

        mu = mu_logvar[
            :,
            :self._latent_dim
        ]

        logvar = mu_logvar[
            :,
            self._latent_dim:
        ]

        return mu, logvar