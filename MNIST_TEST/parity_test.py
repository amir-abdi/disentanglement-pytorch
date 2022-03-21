import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from mlp_models import VAE, CBM


def vae_train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = data.cuda()
        y = y.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var, z = vae(data)
        pred, _ = vae.predict(z)

        loss, _ = vae.loss_function(recon_batch, data, mu, log_var, z, pred, y)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def vae_test():
    vae.eval()
    test_loss = 0
    test_dict = {'rec': 0, 'kld': 0, 'pred': 0, 'lat': 0}
    with torch.no_grad():
        for data, y in test_loader:
            data = data.cuda()
            y = y.cuda()
            recon, mu, log_var, z = vae(data)
            pred, _ = vae.predict(z)

            # sum up batch loss
            p_loss, l_dict = loss_function(recon, data, mu, log_var, z, pred, y)
            test_dict['rec'] += l_dict['rec'].item() / len(test_loader.dataset)
            test_dict['kld'] += l_dict['kld'].item() / len(test_loader.dataset)
            test_dict['lat'] += l_dict['lat'].item() / len(test_loader.dataset)
            test_dict['pred'] += l_dict['pred'].item() / len(test_loader.dataset)
            test_loss += p_loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print(test_dict)


def cbm_train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = data.cuda()
        y = y.cuda()
        optimizer.zero_grad()

        mu = cbm(data)
        pred, _ = cbm.predict(mu)

        loss, _ = cbm.loss_function(data, mu, pred, y, version='join')

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def cbm_test():
    cbm.eval()
    test_loss = 0
    test_dict = {'rec': 0, 'kld': 0, 'pred': 0, 'lat': 0}
    with torch.no_grad():
        for data, y in test_loader:
            data = data.cuda()
            y = y.cuda()
            mu = cbm(data)
            pred, _ = cbm.predict(mu)

            # sum up batch loss
            p_loss, l_dict = cbm.loss_function(data, mu, pred, y, version='join')

            test_dict['lat'] += l_dict['lat'].item() / len(test_loader.dataset)
            test_dict['pred'] += l_dict['pred'].item() / len(test_loader.dataset)
            test_loss += p_loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print(test_dict)


def train_parity(epoch, model, optimizer, dataloader):
    model.train()

    train_loss = 0
    for batch_idx, (data,y) in dataloader:


        data = data.cuda()
        y = y.cuda()

        optimizer.zero_grad()

        mu = model(data)
        pred, _ = model.predict(mu)

        loss, _ = model.loss_function(data, mu, pred, y, only_class=True)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))

    print('TRAIN COMPLETED')
    return model

def test_parity(model, test_loader):
    bs = test_loader.batch_size
    accuracy = 0
    for index, (x,y) in enumerate(test_loader):
        _, prob = model.predict(x[:, :2])

        for j, p in enumerate(prob):
            if torch.argmax(p) == y[j]: accuracy += 1/bs
    print('Overall performance:', accuracy / bs)
    return accuracy / bs

def information_leakage():

    ## INITIALIZATION
    bs = 100
    # MNIST Dataset
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [40000, 20000])

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    print('# Start parity test. First with VAE model')
    ### BUILD VAE
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=5)
    if torch.cuda.is_available():
        vae.cuda()
        print('CUDA ON')
    optimizer = optim.Adam(vae.parameters())

    for epoch in range(51):
        vae_train(epoch)
        vae_test()

    del optimizer

    ### PARITY TEST ###
    vae_optim = optim.SGD(vae.classifier.parameters(),
                          lr = 0.01,
                          momentum=0.9)

    for epoch in range(10):
        vae = train_parity(epoch, vae, vae_optim, val_loader)

    vae_accuracy = test_parity(vae, test_loader)


    print('# Start parity test with CBM model')

    cbm = CBM(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
    if torch.cuda.is_available():
        cbm.cuda()
        print('CUDA ON')

    optimizer = optim.Adam(cbm.parameters())

    for epoch in range(51):
        cbm_train(epoch)
        cbm_test()

    del optimizer

    ### PARITY TEST ###
    cbm_optim = optim.SGD(cbm.classifier.parameters(),
                          lr=0.01,
                          momentum=0.9)
    for epoch in range(10):
        cbm = train_parity(epoch, cbm, cbm_optim, val_loader)

    cbm_accuracy = test_parity(cbm, test_loader)
