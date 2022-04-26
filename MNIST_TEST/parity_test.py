from grpc import method_handlers_generic_handler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from mlp_models import VAE, CBM, OSR_VAE
from test_functions import *

import os

def information_leakage():

    ## INITIALIZATION
    bs = 100
    # MNIST Dataset
    
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)
    #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [40000, 20000])

    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)


    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    val_loader = train_loader #torch.utils.data.DataLoader(dataset=val_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


    path = '/home/emanuele.marconato/disentanglement-pytorch/mnist_data/'


    ### BUILD VAE
    print('# Start parity test. First with VAE model')
    
    if os.path.exists(path+'vae.pt'):
        print('Charged existing vae.pt')
        vae = torch.load(path+'vae.pt')
    else:
        vae = VAE(x_dim=784, h_dim1=128, h_dim2=128, z_dim=10, beta=5)
        if torch.cuda.is_available(): vae.cuda()
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
#        lr_scheduler = optim.lr_scheduler.ExponentialLR() (optimizer, 0.75)
        
        vae = vae_train(vae, optimizer, train_loader, test_loader)
        
          
        with torch.no_grad():
            z = torch.randn(size=(64, 10)).cuda()
            sample = vae.decoder(z).cuda()
            save_image(sample.view(64, 1, 28, 28), path+'samples/sample_MNIST' + '.png')
        
        del optimizer


        torch.save(vae, path+'vae.pt')

    ### PARITY TEST ###
    '''
    vae_optim = optim.SGD(vae.classifier.parameters(),
                          lr = 0.01,
                          momentum=0.1)

    for epoch in range(3):
        print('VAE, epoch:', epoch+1)
        vae = train_parity(epoch, vae, vae_optim, val_loader, name='vae')

        vae_accuracy = test_parity(vae, test_loader)
    '''

    
    ### BUILD CBM
    print('# Start parity test with CBM model')

    if os.path.exists(path+'cbm.pt'):
        print('Charged existing cbm.pt')
        cbm = torch.load(path+'cbm.pt')
    else:
        cbm = CBM(x_dim=784, h_dim1= 128, h_dim2=128, z_dim=2)
        if torch.cuda.is_available():
            cbm.cuda()

        optimizer = optim.Adam(cbm.parameters())
        cbm = cbm_train(cbm, optimizer, train_loader, test_loader)
        
        del optimizer
        
        torch.save(cbm, path+'cbm.pt')


    ### PARITY TEST ###
    '''
    cbm_optim = optim.SGD(cbm.classifier.parameters(),
                          lr=0.01,
                          momentum=0.9)
    for epoch in range(3):
        print('CBM, epoch:', epoch+1)

        cbm = train_parity(epoch, cbm, cbm_optim, val_loader, name='cbm')

        cbm_accuracy = test_parity(cbm, test_loader, name='cbm')
    '''

    ### BUILD OSR_VAE
    print('# Start parity test. Now with OSR_VAE model')
    
    if os.path.exists(path+'osr_vae.pt'):
        print('Charged existing osr_vae.pt')
        osr_vae = torch.load(path+'osr_vae.pt')
        y_hot = torch.tensor([[1,0],[0,1]], dtype=torch.float).cuda()
        print('Encoding for 4 and 5:')
        print(osr_vae.z_enc(y_hot))
        
        # CREATE RECONS
        with torch.no_grad():
            y = torch.randint(2, size=(64,)).cuda()
            y_hot = F.one_hot(y, num_classes=2).cuda()
            mu_y = torch.zeros(size=(64,10)).cuda()
            mu_y[:,:2] = osr_vae.z_enc(y_hot.to(torch.float))
            z =  mu_y + torch.normal(0, 1, size=mu_y.size()).cuda()
            #z = torch.randn( size=(64, 10)).cuda()
            sample = osr_vae.decoder(z).cuda()
            save_image(sample.view(64, 1, 28, 28), path+'samples/osr_sample_MNIST' + '.png')
    else:
        osr_vae = OSR_VAE(x_dim=784,h_dim1=128, h_dim2=128, z_dim=10, beta=5)
        if torch.cuda.is_available(): osr_vae.cuda()
        optimizer = optim.Adam(osr_vae.parameters())

        osr_vae = osr_vae_train(osr_vae, optimizer, train_loader, test_loader)
        osr_vae_test(osr_vae, test_loader)
        # END OF EPOCH
        
        y_hot = torch.tensor([[1,0],[0,1]], dtype=torch.float).cuda()
        print('Encoding for 4 and 5:')
        print(osr_vae.z_enc(y_hot))
        
        ## CREATE RECONS
        with torch.no_grad():
            y = torch.randint(2, size=(64,)).cuda()
            y_hot = F.one_hot(y, num_classes=2).cuda()
            mu_y = torch.zeros(size=(64,10)).cuda()
            mu_y[:,:2] = osr_vae.z_enc(y_hot.to(torch.float))
            z =  mu_y + torch.normal(0, 1, size=mu_y.size()).cuda()
            #z = torch.randn( size=(64, 10)).cuda()
            sample = osr_vae.decoder(z).cuda()
            save_image(sample.view(64, 1, 28, 28), path+'samples/osr_sample_MNIST' + '.png')
        
        del optimizer


        torch.save(osr_vae, path+'osr_vae.pt')

if __name__=='__main__':
    information_leakage()