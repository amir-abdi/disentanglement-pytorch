import torch.nn as nn
import torch.nn.init as init


class BaseImageDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.image_size = image_size

    def init_layers(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    init.xavier_normal_(m.weight.data)
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight.data)

    def forward(self, *input):
        raise NotImplementedError
