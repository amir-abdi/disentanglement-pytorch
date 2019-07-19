__doc__ = """
Example training script with PyTorch. Here's what you need to do. 

Before you run this script, ensure that the following environment variables are set:
    1. AICROWD_OUTPUT_PATH (default: './scratch/shared')
    2. AICROWD_EVALUATION_NAME (default: 'experiment_name')
    3. AICROWD_DATASET_NAME (default: 'cars3d')
    4. DISENTANGLEMENT_LIB_DATA (you may set this to './scratch/dataset' if that's 
                                 where the data lives)

We provide utility functions to make the data and model logistics painless. 
But this assumes that you have set the above variables correctly.    

Once you're done with training, you'll need to export the function that returns
the representations (which we evaluate). This function should take as an input a batch of 
images (NCHW) and return a batch of vectors (NC), where N is the batch-size, C is the 
number of channels, H and W are height and width respectively. 

To help you with that, we provide an `export_model` function in utils_pytorch.py. If your 
representation function is a torch.jit.ScriptModule, you're all set 
(just call `export_model(model)`); if not, it will be traced (!) and the resulting ScriptModule 
will be written out. To learn what tracing entails: 
https://pytorch.org/docs/stable/jit.html#torch.jit.trace 

You'll find a few more utility functions in utils_pytorch.py for pytorch related stuff and 
for data logistics.
"""

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import utils_pytorch as pyu

import aicrowd_helpers

parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = pyu.get_loader(batch_size=args.batch_size, **kwargs)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.tail = nn.Sequential(nn.Linear(4096 * 3, 400),
                                  nn.ReLU())
        self.head_mu = nn.Linear(400, 20)
        self.head_logvar = nn.Linear(400, 20)

    def forward(self, x):
        h = self.tail(x.contiguous().view(-1, 4096 * 3))
        return self.head_mu(h), self.head_logvar(h)


class Decoder(nn.Sequential):
    def __init__(self):
        super(Decoder, self).__init__(nn.Linear(20, 400),
                                      nn.ReLU(),
                                      nn.Linear(400, 4096 * 3),
                                      nn.Sigmoid())


class RepresentationExtractor(nn.Module):
    VALID_MODES = ['mean', 'sample']

    def __init__(self, encoder, mode='mean'):
        super(RepresentationExtractor, self).__init__()
        assert mode in self.VALID_MODES, '`mode` must be one of {self.VALID_MODES}'
        self.encoder = encoder
        self.mode = mode

    def forward(self, x):
        mu, logvar = self.encoder(x)
        if self.mode == 'mean':
            return mu
        elif self.mode == 'sample':
            return self.reparameterize(mu, logvar)
        else:
            raise NotImplementedError

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = RepresentationExtractor.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4096 * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).float()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


if __name__ == '__main__':
    # Go!
    aicrowd_helpers.execution_start()
    aicrowd_helpers.register_progress(0.)
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        aicrowd_helpers.register_progress(epoch / (args.epochs + 1))
    # Almost done...
    aicrowd_helpers.register_progress(0.90)
    # Export the representation extractor
    path_to_saved = pyu.export_model(RepresentationExtractor(model.encoder, 'mean'),
                                     input_shape=(1, 3, 64, 64))
    # Done!
    aicrowd_helpers.register_progress(1.0)
    aicrowd_helpers.submit()
