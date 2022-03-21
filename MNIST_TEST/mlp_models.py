import torch
import torch.nn as nn
import torch.nn.functional as F

def pred_loss(pred, ys):
    loss = 0
    for i, y in enumerate(ys):
        val = torch.zeros(size=ys.size())
        if y % 2 == 0:
            val[i] = 0
        else:
            val[i] = 1

        if y != 4 and y != 5:
            loss += nn.CrossEntropyLoss(reduction='sum')(pred, val.to(dtype=torch.long).cuda())

    return loss

def latent_error(z, ys):
    z_true = torch.zeros(size=(len(z), 2))
    for i, y in enumerate(ys):
        if y == 4:
            z_true[i, 0] = 1
        if y == 5:
            z_true[i, 1] = 1
    z_true = z_true.cuda()
    z = torch.sigmoid(z)
    return nn.MSELoss(reduction='sum')(z[:,:2], z_true)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

        # classifier
        self.classifier = nn.Linear(2, 2)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def predict(self, z):
        prob = nn.Softmax()(z)[:, 1]
        return self.classifier(z), prob

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var, z=None, pred=None, y=None, only_class=False ):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        ## ADD 4/5 test for parity prediction

        PRED, LAT = 0, 0
        if y is not None:
            PRED = pred_loss(pred, y) * 100

            LAT = latent_error(z, y) * 100

        if only_class:
            return PRED, {'pred': PRED}
        else:
            return BCE + KLD + LAT, {'rec': BCE, 'kld': KLD, 'pred': PRED, 'lat': LAT}  # PRED + LAT


class CBM(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, version='seq'):
        super(CBM, self).__init__()

        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, z_dim)

        # classifier
        self.classifier = nn.Linear(z_dim, 2)

        # cbm type
        self.version = version

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def predict(self, z):
        prob = nn.Softmax()(self.classifier(z))[:, 1]
        return self.classifier(z), prob

    def forward(self, x):
        return self.encoder(x.view(-1, 784))

    def loss_function(self, x, mu, pred, y, mse_on_z=True, only_class=False):

        ## ADD 4/5 test for parity prediction
        PRED, LAT = 0, 0
        PRED = pred_loss(pred, y) * 100

        if only_class:
            return PRED, {'pred': PRED}

        if mse_on_z: LAT = latent_error(mu, y) * 100

        # TYPE OF CBMS
        if self.version == 'seq':
            return LAT, {'pred': PRED, 'lat': LAT}  # PRED + LAT

        elif self.version == 'join':
            return PRED + LAT, {'pred': PRED, 'lat': LAT}  # PRED + LAT
        else:
            NotImplementedError('Wrong choice!')