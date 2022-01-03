import torch
from torch import nn
import torch.optim as optim

from models.vae import VAE
from architectures import encoders, decoders, others
from common.ops import reparametrize
from common.utils import one_hot_embedding
from common import constants as c

class GRAYVAEModel(nn.Module):
    def __init__(self, encoder, decoder, tiler, num_classes):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tiler = tiler
        self.num_classes = num_classes

    def encode(self, x, c):
        """
        :param x: input data
        :param c: labels with dtype=long, where the number indicates the class of the input (i.e. not one-hot-encoded)
        :return: latent encoding of the input and labels
        """
        y_onehot = one_hot_embedding(c, self.num_classes).squeeze(1)
        y_tiled = self.tiler(y_onehot)
#        print("One_hot embedding in encoder", y_tiled.size())
        xy = torch.cat((x, y_tiled), dim=1)
        return self.encoder(xy)

    def decode(self, z, c):
        """

        :param z: latent vector
        :param c: labels with dtype=long, where the number indicates the class of the input (i.e. not one-hot-encoded)
        :return: reconstructed data
        """
        y_onehot = one_hot_embedding(c, self.num_classes).squeeze(1)
        zy = torch.cat((z, y_onehot), dim=1)
        return torch.sigmoid(self.decoder(zy))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)

        z = reparametrize(mu, logvar)
        return self.decode(z, c)


class GRAYVAE(VAE):
    """
    Graybox version of VAE. The discussion on
    """

    def __init__(self, args):
        super().__init__(args)

        # checks
        assert self.num_classes is not None, 'please identify the number of classes for each label separated by comma'

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]
        label_tiler_name = args.label_tiler[0]

        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)
        tile_network = getattr(others, label_tiler_name)

        # number of channels
        image_channels = self.num_channels
        label_channels = sum(self.num_classes)
        input_channels = image_channels + label_channels
        decoder_input_channels = self.z_dim + label_channels

        # model and optimizer
        self.model = GRAYVAEModel(encoder(self.z_dim, input_channels, self.image_size),
                               decoder(decoder_input_channels, self.num_channels, self.image_size),
                               tile_network(label_channels, self.image_size),
                               self.num_classes).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # update nets
        self.net_dict['G'] = self.model
        self.optim_dict['optim_G'] = self.optim_G

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

        ## add binary classification layer
        self.classification = nn.Linear(self.z_dim, 1, bias=False).to(self.device)

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        labels = kwargs['labels']
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        mu, logvar = self.model.encode(x=images, c=labels)
        return mu

    def decode(self, **kwargs):
        latent = kwargs['latent']
        labels = kwargs['labels']
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        return self.model.decode(z=latent, c=labels)

    def predict(self, **kwargs):
        """
        Predict the correct class for the input data.
        """
        input_x = kwargs['latent'].to(self.device)
        #print("predict term",input_x)
        return nn.Sigmoid()(self.classification(input_x).resize(len(input_x)))

    def vae_classification(self, losses, x_true1, label1, true_labels, y_true1, labelling=False):
        mu, logvar = self.model.encode(x=x_true1, c=label1)
        #print("Mu", mu.size())
        #print("mu", mu.size())
        z = reparametrize(mu, logvar)
        x_recon = self.model.decode(z=z, c=label1)
        #added here
        prediction = self.predict(latent=mu)

        loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
        losses.update(self.loss_fn(losses, **loss_fn_args))
        # add the classification loss
        #print("label",true_labels[0])
        #print("y_true", (y_true1[:10]))
        #print("prediction", (prediction[:10]))
        losses.update(prediction=nn.BCEWithLogitsLoss()(prediction,y_true1.to(self.device, dtype= torch.float) ))

        if labelling:
            z_real = z[:, :true_labels.size(1)]
#            print("Len z and true", z_real.size(), true_labels.size())
            losses.update(true_values=nn.MSELoss()(z_real, true_labels))
        #print("BCE loss of classification",nn.BCEWithLogitsLoss()(prediction,y_true1.type(torch.FloatTensor)))

        return losses, {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar, "prediction": prediction}


    def train(self):
        while not self.training_complete():
            self.net_mode(train=True)
            vae_loss_sum = 0
#            print("Inside data_loader")
#            print(next(iter(self.data_loader)))
            printme = False
            for internal_iter, (x_true1, label1, z) in enumerate(self.data_loader):
                if printme:
                    print("x_true", x_true1.size(), "label", label1.size())
                    print("z:", z[:10])
                losses = dict()
                x_true1 = x_true1.to(self.device)
                label1 = label1.to(self.device)
                z = z.to(self.device)
                ## loader here
                y_true1 = next(iter(self.target_loader))

                losses, params = self.vae_classification(losses, x_true1, label1, z, y_true1, labelling=True)

                self.optim_G.zero_grad()

                if (internal_iter%100)==0: print("Losses:", losses)

                losses[c.TOTAL_VAE].backward(retain_graph=False)
                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum / internal_iter

                self.optim_G.step()
                self.log_save(input_image=x_true1, recon_image=params['x_recon'], loss=losses)
            # end of epoch
        self.pbar.close()
