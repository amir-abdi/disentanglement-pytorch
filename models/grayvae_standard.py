import copy
import os.path

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from models.vae import VAE
from models.vae import VAEModel
from architectures import encoders, decoders
from common.ops import reparametrize
from common.utils import F1_Loss, Accuracy_Loss
from common import constants as c
from common.data_loader import  target_cast

from sklearn.linear_model import LogisticRegression

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
        self.optim_G_mse = optim.Adam(self.model.encoder.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

#        print("Printing parameters")
 #       for param in self.model.parameters():
  #          print(type(param), param.size())

   #     quit()
        # update nets
        self.net_dict['G'] = self.model
        self.optim_dict['optim_G'] = self.optim_G

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

        ## add binary classification layer
#        self.classification = nn.Linear(self.z_dim, 1, bias=False).to(self.device)
        self.classification = nn.Linear(self.z_dim, 2, bias=False).to(self.device) ### CHANGED OUT DIMENSION
        self.classification_epoch = args.classification_epoch
        self.reduce_rec = args.reduce_recon

        self.class_G = optim.SGD(self.classification.parameters(), lr=0.01, momentum=0.9)

    def predict(self, **kwargs):
        """
        Predict the correct class for the input data.
        """
        input_x = kwargs['latent'].to(self.device)
        pred = nn.Softmax(dim=1)( self.classification(input_x) ,)
        return  pred.to(self.device, dtype=torch.float32) #nn.Sigmoid()(self.classification(input_x).resize(len(input_x)))

    def vae_classification(self, losses, x_true1, label1, y_true1, labelling=False):
        #x_true1.requires_grad = True
        mu, logvar = self.model.encode(x=x_true1,)

        z = reparametrize(mu, logvar)
        x_recon = self.model.decode(z=z,)

        # CHECKING THE CONSISTENCY
        z_prediction = torch.zeros(size=(len(mu), self.z_dim))
        z_prediction[:, :5] = label1
        z_prediction[:, 5:] = mu[:, 5:]

        mu_detatch = z_prediction

        prediction = self.predict(latent=mu_detatch)

        if not labelling:
            loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
            loss_dict = self.loss_fn(losses, reduce_rec=labelling, **loss_fn_args)

            losses.update({'total_vae': loss_dict['total_vae'].detach(), 'recon': loss_dict['recon'].detach(),
                           'kld': loss_dict['kld'].detach()})

            del loss_dict
            ## REMOVED FOR SIGNAL
#            chosen_value=2
            losses.update(true_values=nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1))
            #losses.update(true_values=nn.MSELoss(reduction='mean')(mu[:, ], label1[:,1:] ))

            losses[c.TOTAL_VAE] += nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1).detach()
#            print("MSE loss of true latents",nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1))
            #print("MSE loss for true latents" , nn.MSELoss(reduction='mean')(mu[:, chosen_value], label1[:,chosen_value] ) )

        if labelling:
            ## DISJOINT VERSION
            #print("Prediction", torch.max(prediction), torch.min(prediction))
            #print("Y TRUE", y_true1[:10])
            loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
            loss_dict = self.loss_fn(losses, reduce_rec=labelling, **loss_fn_args)
            loss_dict.update(true_values=nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1))
            loss_dict[c.TOTAL_VAE] += nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1)
            losses.update({'total_vae': loss_dict['total_vae'].detach(), 'recon': loss_dict['recon'].detach(),
                           'kld': loss_dict['kld'].detach(), 'true_values': loss_dict['true_values'].detach()})

            del loss_dict

            losses.update(prediction=nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1.to(self.device, dtype=torch.long)))
            #losses[c.TOTAL_VAE] += nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1.to(self.device, dtype=torch.long))

            #losses.update(prediction=nn.BCEWithLogitsLoss()(prediction, y_true1.to(self.device, dtype=torch.float)))
            #losses[c.TOTAL_VAE] = nn.BCEWithLogitsLoss()(prediction,y_true1.to(self.device, dtype= torch.float))

            ## INSERT DEVICE IN THE CREATION OF EACH TENSOR
            ### AVOID COPYING FROM CPU TO GPU AS MUCH AS POSSIBLE

            """ 
            Random = [0.54736527, 0.22488107, 0.17828586, 0.4332863, 0.56544113, 0.33252552]
            Mean = [1., 2.00356, 0.7513, 3.14917759, 0.49945935, 0.49974452]
            r_plane = [Random, Mean]

            print("Z latent values")
            print(label1[:10])# -torch.tensor(Mean[1:]))
            print("Len", len(label1))

            z_prediction_smaller = z_prediction[:, :5]

            y_real = target_cast( z_prediction_smaller.detach().numpy(),  r_plane, irrelevant_components=0)
            y_real = torch.tensor(y_real, device=self.device, dtype=torch.long)
            ### y_real is the real value as predicted by r_plane

            print("REALVALUES - The percentage of 0s is ", len(y_real[y_real==0])/len(y_real)*100, "%" )
            print("PASSEDVALUES - The percentage of 0s is ", len(y_real[y_true1 == 0]) / len(y_real) * 100, "%")

            in_data = z_prediction
            lr = LogisticRegression()
            lr.fit(in_data.detach(), y_true1)
            y_pred1 = lr.predict_proba(in_data.detach())
            y_pred1 = torch.tensor(y_pred1, dtype=torch.float)
            ### y_pre1 is the prediction from LogReg
            """
#            print("SKLEARN/REAL ", torch.mean((y_pred1[:,1] -y_real)**2))
 #           print(" ")
            #print("Analytic loss:", nn.BCEWithLogitsLoss()(y_pred1[:,1],y_true1.to(self.device, dtype= torch.float)).item() )
            #print("Analytic loss from y_real:", nn.BCEWithLogitsLoss()(prediction, y_real).item() )
            #print('Difference between y_real and y_loaded', nn.BCEWithLogitsLoss()(y_real,y_true1.to(self.device, dtype= torch.float)).item() )
            #print("Loss from y_real",  nn.BCEWithLogitsLoss()(y_pred1[:,0], y_real).item() )


            #print("CrossEntropy", nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1))
#            print("CrossEntropy for y_pred from LOG-REG ", nn.CrossEntropyLoss(reduction='mean')(y_pred1, y_true1))


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

        #global chosen_value
        #chosen_value = 1

        while not self.training_complete():
            Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores = [], [], [], [], [], [], [] ## JUST HERE FOR NOW

            epoch += 1
            self.net_mode(train=True)
            vae_loss_sum = 0
            # add the classification layer #
            if epoch>self.classification_epoch:
                print("## STARTING CLASSIFICATION ##")
                start_classification = True
            else: start_classification = False

            for internal_iter, (x_true1, label1, y_true1) in enumerate(self.data_loader):
                Iterations.append(internal_iter+1)
                Epochs.append(epoch)
                losses = {'total_vae':0}

                x_true1 = x_true1.to(self.device)
                label1 = label1[:, 1:].to(self.device)
                y_true1 = y_true1.to(self.device)

                #for i in label1[:, chosen_value]:
                 #   collection.append(i)

                #plt.hist(collection, bins=int(len(collection)/10), )
                #plt.show()

                ###configuration for dsprites

                losses, params = self.vae_classification(losses, x_true1, label1, y_true1,
                                                         labelling=start_classification)

                self.optim_G.zero_grad()
                self.optim_G_mse.zero_grad()
                self.class_G.zero_grad()

                if (internal_iter%250)==0: print("Losses:", losses)

                if not start_classification:
                    #losses[c.TOTAL_VAE].backward(retain_graph=False)
                    losses['true_values'].backward(retain_graph=False)
                    self.optim_G.step()
                    """ 
                    print("GRADIENTS")
                    print((self.model.encoder.main[-1].weight.grad))
                    print(" ")
                    print(" ")
                    print(" SIZE OF GRADIENT ")
                    print(self.model.encoder.main[-1].weight.size())
                    """
                    #losses['true_values'].backward(retain_graph=False)
                    #self.optim_G_mse.step()

                #                print("Weights")
 #               past_weights = self.model.decoder.main[1].weight.grad
                #print(self.model.decoder.main[1].weight.grad[:4])

                ## SWITCH OFF PROP OF GRADIENT
#                for parameters in self.model.parameters():
 #                   parameters.requires_grad = False

                ## INSERT HERE CLASSIFICATION
                if start_classification:

                    losses['prediction'].backward(retain_graph=False)
                    #print('Weight of classification:',self.classification.weight)
                    #print('Gradients for classification', self.classification.weight.grad )
                    #print("The optimizer")
                    #print(self.class_G)
                    self.class_G.step()

#                print("Weights")
 #               print("Changed of", torch.mean(self.model.decoder.main[1].weight.grad -past_weights))
                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum /( internal_iter+1) ## ADDED +1 HERE IDK WHY NOT BEFORE!!!!!

                ## Insert losses -- only in training set
                if track_changes:

                    rec_err = losses['recon'].item()
                    Reconstructions.append(rec_err)
                    kld_error = losses['kld'].item()
                    KLDs.append(kld_error)
                    mse_true = losses['true_values'].item()
                    True_Values.append(mse_true)
                    if not start_classification: #RECONSTRUCTION ERROR + KLD + MSE on Z
                        Accuracies, F1_scores = [0] * len(Iterations), [0] * len(Iterations)

                    else: #CLASSIFICATION

                        accuracy = losses['prediction']
                        Accuracies.append(accuracy.item())

                        F1_scores = [0] * len(Iterations)
                        #f1_class = F1_Loss().to(self.device)
                        #F1_scores.append(f1_class(y_pred1, y_true1).item())

                    sofar = pd.DataFrame(data=np.array([Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores]).T,
                                         columns=['iter', 'epoch', 'reconstruction_error', 'kld', 'latent_error', 'accuracy', 'f1_score'], )

                    sofar.to_csv(os.path.join(out_path, 'metrics.csv'), index=False)
                self.log_save(input_image=x_true1, recon_image=params['x_recon'], loss=losses)
            #chosen_value += 1
            #if chosen_value == 5: break
            """ 
            #insert into pd dataframe
            if track_changes:
                if not start_classification:
                    Accuracies, F1_scores = [0] * len(Iterations), [0] * len(Iterations)
                else:
                    Reconstructions, KLDs, True_Values = [0]*len(Iterations), [0]*len(Iterations), [0]*len(Iterations)

                sofar = pd.DataFrame(data=np.array([Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores]).T,
                                     columns=['iter', 'epoch', 'reconstruction_error', 'kld', 'latent_error', 'accuracy', 'f1_score'], )
                sofar.to_csv(os.path.join(out_path,'metrics.csv'), index=False)

            # end of epoch
            """

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

