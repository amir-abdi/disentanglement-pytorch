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

        self.label_weight = args.label_weight
        self.masking_fact = args.masking_fact

        self.dataframe_eval = pd.DataFrame(columns=self.evaluation_metric)

    def predict(self, **kwargs):
        """
        Predict the correct class for the input data.
        """
        input_x = kwargs['latent'].to(self.device)
        pred_raw = self.classification(input_x)
        pred = nn.Softmax(dim=1)(pred_raw)
        return  pred_raw, pred.to(self.device, dtype=torch.float32) #nn.Sigmoid()(self.classification(input_x).resize(len(input_x)))

    def vae_classification(self, losses, x_true1, label1, y_true1, classification=False):
        #x_true1.requires_grad = True
        mu, logvar = self.model.encode(x=x_true1,)

#        z = reparametrize(mu, logvar)
        z = torch.tanh(2*reparametrize(mu, logvar))
        x_recon = self.model.decode(z=z,)

        # CHECKING THE CONSISTENCY
        z_prediction = torch.zeros(size=(len(mu), self.z_dim))
        z_prediction[:, :label1.size(1)] = label1
        z_prediction[:, label1.size(1):] = mu[:, label1.size(1):].detach()

        prediction, _ = self.predict(latent=z_prediction)

        if not classification:
            loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
            loss_dict = self.loss_fn(losses, reduce_rec=False, **loss_fn_args)
            losses.update(loss_dict)
#            losses.update({'total_vae': loss_dict['total_vae'].detach(), 'recon': loss_dict['recon'].detach(),
 #                          'kld': loss_dict['kld'].detach()})
            del loss_dict
            ## REMOVED FOR SIGNAL
#            chosen_value=2
#            losses.update(true_values=nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1))
#            loss_soft = nn.CrossEntropyLoss(reduction='mean')(mu[:,:3], label1[:,:3])

            rn_mask = (torch.randn(size=(self.batch_size,)) < self.masking_fact/100*torch.ones(size=(self.batch_size,)) )
            if len(z[rn_mask]) > 0: # added the presence of only small labelled generative factors
                ## loss of categorical variables
                #loss_bin =  nn.BCELoss(reduction='mean')( (1+z[rn_mask][:,:3])/2, label1[rn_mask][:,:3])*(3/label1.size(1))
                ## loss of continuous variables
                loss_bin = nn.MSELoss(reduction='mean')( z[rn_mask][:, :label1.size(1)], 2*label1[rn_mask]-1  ) #*(1 - 3/label1.size(1))

                ## track losses
                err_latent = []
                for i in range(label1.size(1)):
                    err_latent.append(nn.MSELoss(reduction='mean')(z[rn_mask][:, i], 2 * label1[rn_mask][:,i] - 1).detach().item() )
                    """ 
                    if i < 3:
                        err_latent.append(nn.BCELoss(reduction='mean')( (1+z[rn_mask][:,i])/2, label1[:,i][rn_mask] ).detach().item())
                    else:
                        err_latent.append(nn.MSELoss(reduction='mean')(z[rn_mask][:,i], 2*label1[rn_mask][:,1] -1).detach().item() )
                    """
                losses.update(true_values= self.label_weight*loss_bin)
                #losses.update(true_values=nn.MSELoss(reduction='mean')(mu[:, ], label1[:,1:] ))
                losses[c.TOTAL_VAE] += self.label_weight*loss_bin.detach()

                ## REMOVE THIS PIECE
                #losses[c.TOTAL_VAE] = losses[c.TOTAL_VAE].detach()
                #losses['recon'], losses['kld'] = torch.tensor(0), torch.tensor(0)

        #            losses[c.TOTAL_VAE] += nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1).detach()

        if classification:
            #TODO: INSERT MORE OPTIONS ON HOW TRAINING METRICS AFFECT
            ## DISJOINT VERSION
            loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
            loss_dict = self.loss_fn(losses, reduce_rec=True, **loss_fn_args)
            loss_dict.update(true_values=nn.BCELoss(reduction='mean')((1+z[:,:label1.size(1)])/2, label1))
            loss_dict[c.TOTAL_VAE] += nn.BCELoss(reduction='mean')((1+z[:, :label1.size(1)])/2, label1)
            losses.update({'total_vae': loss_dict['total_vae'].detach(), 'recon': loss_dict['recon'].detach(),
                           'kld': loss_dict['kld'].detach(), 'true_values': loss_dict['true_values'].detach()})

            del loss_dict
            #TODO insert BCE Classification

            losses.update(prediction=nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1.to(self.device, dtype=torch.long)))
            #losses[c.TOTAL_VAE] += nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1.to(self.device, dtype=torch.long))
            #losses.update(prediction=nn.BCEWithLogitsLoss()(prediction, y_true1.to(self.device, dtype=torch.float)))
            #losses[c.TOTAL_VAE] = nn.BCEWithLogitsLoss()(prediction,y_true1.to(self.device, dtype= torch.float))

            ## INSERT DEVICE IN THE CREATION OF EACH TENSOR
            ### AVOID COPYING FROM CPU TO GPU AS MUCH AS POSSIBLE

        return losses, {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar, "prediction": prediction,
                        'latents': err_latent}

    def train(self, **kwargs):

        if 'output'  in kwargs.keys():
            out_path = kwargs['output']
            track_changes=True

        else: track_changes=False;

        if track_changes:
            print("## Initializing Train indexes")
            print("::path chosen ->",out_path+"/train_runs")


        #global chosen_value
        #chosen_value = 1

        print("The model we are using")
        print(self.model)
        print('Total parameters:', sum(p.numel() for p in self.model.parameters()))

        Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores = [], [], [], [], [], [], []  ## JUST HERE FOR NOW
        latent_errors = []
        epoch = 0
        while not self.training_complete():

            epoch += 1
            self.net_mode(train=True)
            vae_loss_sum = 0
            # add the classification layer #
            if epoch>self.classification_epoch:
                print("## STARTING CLASSIFICATION ##")
                start_classification = True
            else: start_classification = False

            for internal_iter, (x_true1, label1, y_true1) in enumerate(self.data_loader):

                if internal_iter > 1 and (internal_iter%(self.evaluate_iter) == 1):
                    self.dataframe_eval = self.dataframe_eval.append(self.evaluate_results,  ignore_index=True)


                Iterations.append(internal_iter+1)
                Epochs.append(epoch)
                losses = {'total_vae':0}

                x_true1 = x_true1.to(self.device)
                label1 = label1[:, 1:].to(self.device)
                y_true1 = y_true1.to(self.device)

                ###configuration for dsprites

                losses, params = self.vae_classification(losses, x_true1, label1, y_true1,
                                                         classification=start_classification)

                self.optim_G.zero_grad()
                self.optim_G_mse.zero_grad()
                self.class_G.zero_grad()

                if (internal_iter%250)==0: print("Losses:", losses)

                if not start_classification:
                    losses[c.TOTAL_VAE].backward(retain_graph=False)
                    #losses['true_values'].backward(retain_graph=False)
                    self.optim_G.step()
                    latent_errors.append(params['latents'])

                if start_classification:
                    losses['prediction'].backward(retain_graph=False)
                    self.class_G.step()

                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum /( internal_iter+1) ## ADDED +1 HERE IDK WHY NOT BEFORE!!!!!

                ## Insert losses -- only in training set
                if track_changes:
                    #TODO: set the tracking at a given iter_number/epoch

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

                        #f1_class = F1_Loss().to(self.device)
                        #F1_scores.append(f1_class(y_pred1, y_true1).item())

                    if (internal_iter%500)==0:
                        sofar = pd.DataFrame(data=np.array([Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies]).T,
                                             columns=['iter', 'epoch', 'reconstruction_error', 'kld', 'latent_error', 'accuracy'], )
                        for i in range(label1.size(1)):
                            sofar['latent%i'%i] = np.asarray(latent_errors)[:,i]

                        sofar.to_csv(os.path.join(out_path, 'metrics.csv'), index=False)

                        if not self.dataframe_eval.empty:
                            self.dataframe_eval.to_csv(os.path.join(out_path, 'dis_metrics.csv'), index=False)

                self.log_save(input_image=x_true1, recon_image=params['x_recon'], loss=losses)
            #chosen_value += 1
            #if chosen_value == 5: break
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

