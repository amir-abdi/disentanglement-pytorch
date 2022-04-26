import torch
import torch.nn as nn
import torch.nn.functional as F

def pred_loss(pred, ys):
    
    for i, y in enumerate(ys):
        val = torch.zeros(size=ys.size())
        if y % 2 == 0:
            val[i] = 0
        else:
            val[i] = 1

        #if y != 4 and y != 5:
    loss = nn.CrossEntropyLoss(reduction='mean')(pred, val.to(dtype=torch.long).cuda())
    #print(loss)
    #print(pred)
    #print(val)
    
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
    return nn.BCELoss(reduction='sum')(z[:,:2], z_true)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1=128, h_dim2=128, z_dim=2, beta=10):
        super(VAE, self).__init__()

        self.beta = beta

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
        self.classifier = nn.Sequential(nn.Linear(2,h_dim2), 
                                        nn.ReLU(),
                                        nn.Linear(h_dim2,h_dim1),
                                        nn.ReLU(),
                                        nn.Linear(h_dim1, 2))
        self.classifier = nn.Linear(2,2)
        self.leak_classifier = nn.Linear(z_dim,2)

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
        z = torch.sigmoid(z)
        prob = nn.Softmax(dim=1)(z)[:, 1]
        return self.classifier(z), prob

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var, z=None, pred=None, y=None, only_class=False ):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * self.beta

        ## ADD 4/5 test for parity prediction

        PRED, LAT = 0, 0
        if y is not None:
            PRED = pred_loss(pred, y)

            LAT = latent_error(z, y) * 500

        if only_class:
            BCE, KLD, LAT = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            return PRED, {'pred': PRED}
        else:
            PRED = torch.tensor(0)
            return BCE + KLD + LAT, {'rec': BCE, 'kld': KLD, 'pred': PRED, 'lat': LAT}  # PRED + LAT


class CBM(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, version='seq'):
        super(CBM, self).__init__()

        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, z_dim)

        # classifier
        self.classifier =  nn.Sequential(nn.Linear(2,h_dim2), 
                                        nn.ReLU(),
                                        nn.Linear(h_dim2,h_dim1),
                                        nn.ReLU(),
                                        nn.Linear(h_dim1, 2))

        self.classifier = nn.Linear(2,2)

        # cbm type
        self.version = version

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def predict(self, z):
        z = torch.sigmoid(z)
        prob = nn.Softmax(dim=1)(self.classifier(z))[:, 1]
        return self.classifier(z), prob

    def forward(self, x):
        return self.encoder(x.view(-1, 784))

    def loss_function(self, x, mu, pred, y, mse_on_z=True, only_class=False):

        ## ADD 4/5 test for parity prediction
        PRED, LAT = 0, 0
        PRED = pred_loss(pred, y) #* 100

        if only_class:
            return PRED, {'pred': PRED}

        if mse_on_z: LAT = latent_error(mu, y) #* 100

        # TYPE OF CBMS
        if self.version == 'seq':
            PRED = torch.tensor(0)
            return LAT, {'pred': PRED, 'lat': LAT}  # PRED + LAT

        elif self.version == 'join':
            PRED, LAT = torch.tensor(0), torch.tensor(0)
            return PRED + LAT, {'pred': PRED, 'lat': LAT}  # PRED + LAT
        else:
            NotImplementedError('Wrong choice!')


class C_OSR_VAE(VAE):
    def __init__(self, x_dim, num_classes, prior=None, h_dim1=128, h_dim2=128, z_dim=2, beta=10):
        super().__init__(x_dim=x_dim, h_dim1=128, h_dim2=128, z_dim=2, beta=10)

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.beta = beta
        if prior is None:
            self.prior_on_classes = [0.5, 0.5]
        else:
            self.prior_on_classes = prior
        self.classes = torch.tensor([*range(num_classes)])

        assert sum(self.prior_on_classes) == 1, self.prior_on_classes


        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.mu = nn.ModuleList()
        self.log_var = nn.ModuleList()
        for i in range(num_classes):
            self.mu.append(nn.Linear(h_dim2, z_dim))
            self.log_var.append(nn.Linear(h_dim2, z_dim))
            #self.fc31 = nn.Linear(h_dim2, z_dim)
            #self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

        # classifier 
        
        self.classifier = nn.Linear(2,2)
        self.leak_classifier = nn.Linear(z_dim,2)

        # prior space
        self.z_enc = nn.Linear(num_classes, z_dim, bias=False)

    def encoder(self, x, y):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        mu = torch.empty(size=(len(y),self.z_dim )).cuda()
        log_var = torch.empty(size=(len(y),self.z_dim)).cuda()
        for n in range(self.num_classes):
            mask = (y==n)
            mu[mask] = self.mu[n](h[mask])
            log_var[mask] = self.log_var[n](h[mask])
        return mu, log_var

    def sampling(self, mu, log_var):
        #mu, log_var = mu[y], log_var[y]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x, y=None):

        if y is None:
            y = torch.randint(num_classes) #TODO: must extrct from a given distribution THIS IS THE  
                                           # p(4) = 0.5 and p(5) = 0.5
        mu, log_var = self.encoder(x.view(-1, 784), y)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var, y, mu_y, z=None, pred=None, only_class=False ):
        
        ## ADD 4/5 test for parity prediction

        PRED, LAT = 0, 0
        
        if only_class:
            PRED = pred_loss(pred, y)
            BCE, KLD, LAT = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            return PRED, {'rec': BCE, 'kld': KLD, 'pred': PRED, 'lat': LAT}
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - (mu - mu_y).pow(2) - log_var.exp()) * self.beta
            PRED = torch.tensor(0)
            LAT = latent_error(z, y) * 100
            return BCE + KLD + LAT, {'rec': BCE, 'kld': KLD, 'pred': PRED, 'lat': LAT}  # PRED + LAT



class OSR_VAE(nn.Module):
    def __init__(self, x_dim, h_dim1=128, h_dim2=128, z_dim=2, beta=10):
        super(OSR_VAE, self).__init__()

        self.beta = beta

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
        self.classifier = nn.Sequential(nn.Linear(2,h_dim2), 
                                        nn.ReLU(),
                                        nn.Linear(h_dim2,h_dim1),
                                        nn.ReLU(),
                                        nn.Linear(h_dim1, 2))
        self.classifier = nn.Linear(2,2)
        self.leak_classifier = nn.Linear(z_dim,2)

        self.z_enc = nn.Linear(2, 2, bias=False)


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
        z = torch.sigmoid(z)
        prob = nn.Softmax(dim=1)(z)[:, 1]
        return self.classifier(z), prob

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var, z=None, pred=None, y=None, only_class=False ):
        y_hot = F.one_hot(y-4, num_classes=2).cuda()

        mu_y = torch.zeros(size=mu.size()).cuda()
        mu_y[:,:2] = self.z_enc(y_hot.to(dtype=torch.float))

        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - (mu-mu_y).pow(2) - log_var.exp()) * self.beta

        ## ADD 4/5 test for parity prediction

        PRED, LAT = 0, 0
        if y is not None:
            PRED = pred_loss(pred, y)

            LAT = latent_error(z, y) * 100  
            POST = latent_error(mu_y, y) * 100

        if only_class:
            BCE, KLD, LAT = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            return PRED, {'pred': PRED}
        else:
            PRED = torch.tensor(0)
            return BCE + KLD + LAT+POST, {'rec': BCE, 'kld': KLD, 'pred': PRED, 'lat': LAT, 'post':POST}  # PRED + LAT
