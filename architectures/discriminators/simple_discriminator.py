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


class SimpleDiscriminatorConv64(SimpleDiscriminator):
    # TODO: test
    def __init__(self, num_channels=3, image_size=64, num_classes=2, num_fc_layers=7, fc_layer_size=1000):
        super().__init__(256, num_classes=num_classes, num_layers=num_fc_layers, layer_size=fc_layer_size)
        assert image_size == 64, 'The SimpleDiscriminatorConv64 architecture is hardcoded for 64x64 images.'

        self.conv_encode = nn.Sequential(
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
            self.Flatten(),
        )

        init_layers(self._modules)

    def forward(self, x):
        encoded = self.conv_encode(x)
        return self.main(encoded)
