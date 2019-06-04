import torch.nn as nn

from common.utils import init_layers


class SimpleDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=7, layer_size=1000):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, layer_size),
            nn.LeakyReLU(0.2, True),
        )

        for i in range(num_layers - 2):
            self.main.add_module(module=nn.Linear(layer_size, layer_size), name='linear' + str(i))
            self.main.add_module(module=nn.LeakyReLU(0.2, True), name='lrelu' + str(i))

        self.main.add_module(module=nn.Linear(layer_size, num_classes), name='output')

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)
