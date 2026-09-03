import torch.nn as nn

from common.utils import init_layers


class PersonConv128x64(nn.Module):
    """
    Decoder for pedestrian images of size 128 x 64.

    Input:
        z -> [batch_size, latent_dim]

    Output:
        reconstruction logits
        -> [batch_size, 3, 128, 64]
    """

    def __init__(
        self,
        latent_dim,
        num_channels=3,
        image_size=None
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_channels = num_channels

        # ----------------------------------------------------
        # Latent vector -> convolutional seed
        # ----------------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(
                latent_dim,
                4096
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # Upsampling path
        #
        # Start:
        # [B, 512, 4, 2]
        # ----------------------------------------------------

        self.main = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # [B, 256, 8, 4]

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # [B, 128, 16, 8]

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # [B, 64, 32, 16]

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # [B, 32, 64, 32]

            nn.ConvTranspose2d(
                32,
                num_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )

            # [B, 3, 128, 64]
        )

        init_layers(
            self._modules
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.view(
            z.size(0),
            512,
            4,
            2
        )

        x = self.main(x)

        return x