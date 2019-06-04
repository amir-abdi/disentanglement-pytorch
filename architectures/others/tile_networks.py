import torch
import torch.nn as nn

from common.ops import Reshape
from common.utils import init_layers


class SingleTo2DChannel(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, image_size * image_size),
            Reshape([1, image_size, image_size])
        )

        init_layers(self._modules)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.main(x)


class MultiTo2DChannel(nn.Module):
    def __init__(self, input_dim, image_size):
        super().__init__()

        self.main = nn.ModuleList()

        for i in range(input_dim):
            self.main.append(SingleTo2DChannel(image_size))

        init_layers(self._modules)

    def forward(self, x):
        result = []
        for i, module in enumerate(self.main):
            result.append(module(x[:, i]))

        result = torch.cat(result, dim=1)
        return result

