import os.path

import torch
from torch import nn
import torch.optim as optim
from models.vae import VAE
from models.vae import VAEModel
from architectures import encoders, decoders
from common.ops import reparametrize
from common.utils import F1_Loss, Accuracy_Loss
from common import constants as c

import numpy as np
import pandas as pd

### INSERTING THE LOG SESSION FOR TENSORBOARD ###
#from torch.utils.tensorboard import FileWriter
###                                           ###

class GrayVAE_Standard(VAE):
    """
    Graybox version of VAE, with standard implementation. The discussion on
    """

    def __init__(self, args):
        super().__init__(args)

        print('Initialized GrayVAE_Standard model')

        # checks
        assert self.num_classes is not None, 'please identify the number of classes for each label separated by comma'

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]

        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # number of channels
        image_channels = self.num_channels
        input_channels = image_channels
        decoder_input_channels = self.z_dim

        # model and optimizer
        self.model = VAEModel(encoder(self.z_dim, input_channels, self.image_size),
                               decoder(decoder_input_channels, self.num_channels, self.image_size),
                               ).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # update nets
        self.net_dict['G'] = self.model
        self.optim_dict['optim_G'] = self.optim_G

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

        ## add binary classification layer
        self.classification = nn.Linear(self.z_dim, 1, bias=False).to(self.device)
        self.classification_epoch = args.classification_epoch
        self.reduce_rec = args.reduce_recon

    def predict(self, **kwargs):
        """
        Predict the correct class for the input data.
        """
        input_x = kwargs['latent'].to(self.device)
        return nn.Sigmoid()(self.classification(input_x).resize(len(input_x)))

    def vae_classification(self, losses, x_true1, label1, y_true1, labelling=False):
        x_true1.requires_grad = True
        with torch.no_grad():
            mu, logvar = self.model.encode(x=x_true1,)

        z = reparametrize(mu, logvar)
        x_recon = self.model.decode(z=z,)

        prediction = self.predict(latent=mu)

        loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
        losses.update(self.loss_fn(losses, reduce_rec=labelling, **loss_fn_args))

        if labelling:
            ## DISJOINT VERSION
            losses.update(prediction=nn.BCEWithLogitsLoss()(prediction, y_true1.to(self.device, dtype=torch.float)))
            #losses[c.TOTAL_VAE] += nn.BCEWithLogitsLoss()(prediction,y_true1.to(self.device, dtype= torch.float))


            ## REMOVED FOR SIGNAL
            z_real = z[:, :label1.size(1)]
            losses.update(true_values=nn.MSELoss()(z_real, label1))
            losses[c.TOTAL_VAE] += nn.MSELoss()(z_real, label1)
            #print("BCE loss of classification",nn.BCEWithLogitsLoss()(prediction,y_true1.type(torch.FloatTensor)))

        return losses, {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar, "prediction": prediction}


    def train(self, **kwargs):
        if 'output'  in kwargs.keys():
            out_path = kwargs['output']
            track_changes=True

        else: track_changes=False;

        if track_changes:
            print("## Initializing Train indexes")
            print("::path chosen ->",out_path+"/train_runs")
        epoch = 0
        Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores = [], [], [], [], [], [], []
        while not self.training_complete():
            epoch += 1
            self.net_mode(train=True)
            vae_loss_sum = 0
            # add the classification layer #
            if epoch>self.classification_epoch:
                print("## STARTING CLASSIFICATION ##")
                start_classification = True
            else: start_classification = False

            for internal_iter, (x_true1, _, label1) in enumerate(self.data_loader):
                Iterations.append(internal_iter+1)
                Epochs.append(epoch)
                losses = dict()
                x_true1 = x_true1.to(self.device)
                label1 = label1.to(self.device)

                y_true1 = next(iter(self.target_loader))

                losses, params = self.vae_classification(losses, x_true1, label1, y_true1,
                                                         labelling=start_classification)

                self.optim_G.zero_grad()

                if (internal_iter%250)==0: print("Losses:", losses)


                losses[c.TOTAL_VAE].backward(retain_graph=False)

                ## INSERT HERE CLASSIFICATION
                losses['prediction'].backward()
                #print("Is it backprop to ")
                #print(self.classification.weight.grad)
                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum / internal_iter


                ## Insert losses -- only in training set
                if track_changes:
                    #RECONSTRUCTION ERROR
                    rec_err = losses['recon'].item()
                    Reconstructions.append(rec_err)
                    kld_error = losses['kld'].item()
                    KLDs.append(kld_error)


                    if start_classification: #CLASSIFICATION + TRUE ON LATENT
                        mse_true = losses['true_values'].item()
                        True_Values.append(mse_true)

                        y_pred1 = torch.zeros(size=(len(y_true1), 2), dtype=torch.float ).to(self.device)
                        y_pred1[:,0] = params['prediction']
                        y_pred1[:,1] = 1- params['prediction']

                        accuracy = Accuracy_Loss().to(self.device)
                        Accuracies.append(accuracy(y_pred1, y_true1).item())

                        f1_class = F1_Loss().to(self.device)
                        F1_scores.append(f1_class(y_pred1, y_true1).item())



                self.optim_G.step()
                self.log_save(input_image=x_true1, recon_image=params['x_recon'], loss=losses)

            #insert into pd dataframe
            if track_changes:
                if not start_classification:
                    True_Values, Accuracies, F1_scores = np.zeros(len(Iterations)), np.zeros(len(Iterations)), np.zeros(len(Iterations))

                sofar = pd.DataFrame(data=np.array([Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores]).T,
                                     columns=['iter', 'epoch', 'reconstruction_error', 'kld', 'latent_error', 'accuracy', 'f1_score'], )
                sofar.to_csv(os.path.join(out_path,'metrics.csv'), index=False)

            # end of epoch
        self.pbar.close()


    def test(self):
        self.net_mode(train=False)
        for x_true, label in self.data_loader:
            x_true = x_true.to(self.device)
            label = label.to(self.device, dtype=torch.long)

            x_recon = self.model(x=x_true, c=label)

            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)

            self.iter += 1
            self.pbar.update(1)

