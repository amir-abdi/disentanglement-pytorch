import os.path
import torch
from torch import nn
import torch.optim as optim
from models.vae import VAE
from models.vae import VAEModel
from architectures import encoders, decoders
from common.ops import reparametrize
from common.utils import Accuracy_Loss, Interpretability
from common import constants as c
import torch.nn.functional as F
from common.utils import is_time_for

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
        self.show_loss = args.show_loss

        self.dataframe_dis = pd.DataFrame() #columns=self.evaluation_metric)
        self.dataframe_eval = pd.DataFrame()

        self.latent_loss = args.latent_loss

    def predict(self, **kwargs):
        """
        Predict the correct class for the input data.
        """
        input_x = kwargs['latent'].to(self.device)
        pred_raw = self.classification(input_x)
        pred = nn.Softmax(dim=1)(pred_raw)
        return  pred_raw, pred.to(self.device, dtype=torch.float32) #nn.Sigmoid()(self.classification(input_x).resize(len(input_x)))

    def vae_classification(self, losses, x_true1, label1, y_true1, examples, classification=False):

        mu, logvar = self.model.encode(x=x_true1,)

        z = reparametrize(mu, logvar)
        mu_processed = torch.tanh(z/2)
        x_recon = self.model.decode(z=z,)

        prediction, forecast = self.predict(latent=mu_processed)
        rn_mask = (examples==1)
        n_passed = len(examples[rn_mask])

        if not classification:
            loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
            loss_dict = self.loss_fn(losses, reduce_rec=False, **loss_fn_args)
            losses.update(loss_dict)
#            losses.update({'total_vae': loss_dict['total_vae'].detach(), 'recon': loss_dict['recon'].detach(),
 #                          'kld': loss_dict['kld'].detach()})
            del loss_dict

            if n_passed > 0: # added the presence of only small labelled generative factors

                ## loss of categorical variables

                ## loss of continuous variables
                if self.latent_loss == 'MSE':
                    #TODO: PLACE ONEHOT ENCODING
                    loss_bin = nn.MSELoss(reduction='mean')( mu_processed[rn_mask][:, :label1.size(1)], 2*label1[rn_mask]-1  )
                    ## track losses
                    err_latent = []
                    for i in range(label1.size(1)):
                        err_latent.append(nn.MSELoss(reduction='mean')(mu_processed[rn_mask][:, i], 2 * label1[rn_mask][:,i] - 1).detach().item() )

                    losses.update(true_values=self.label_weight * loss_bin)
                    losses[c.TOTAL_VAE] += self.label_weight * loss_bin

                elif self.latent_loss == 'BCE':

                    loss_bin = nn.BCELoss(reduction='mean')((1+mu_processed[rn_mask][:, :label1.size(1)])/2,
                                                             label1[rn_mask] )

                    ## track losses
                    err_latent = []
                    for i in range(label1.size(1)):
                        err_latent.append(nn.BCELoss(reduction='mean')((1+mu_processed[rn_mask][:, i])/2,
                                                                        label1[rn_mask][:, i] ).detach().item())
                    losses.update(true_values=self.label_weight * loss_bin)
                    losses[c.TOTAL_VAE] += self.label_weight * loss_bin

                elif self.latent_loss == 'exact_BCE':
                    mu_processed = nn.Sigmoid()( mu/torch.sqrt(1+ torch.exp(logvar)) )
                    loss_bin = nn.BCELoss(reduction='mean')( mu_processed[rn_mask], label1[rn_mask] )

                    err_latent = []
                    for i in range(label1.size(1)):
                        err_latent.append( nn.BCELoss(reduction='mean')( mu_processed[rn_mask], label1[rn_mask] ) )

                    losses.update(true_values=self.label_weight * loss_bin)
                    losses[c.TOTAL_VAE] += self.label_weight * loss_bin

                else:
                    raise NotImplementedError('Not implemented loss.')

            else:
                losses.update(true_values=torch.tensor(-1))
                err_latent =[-1]*label1.size(1)
        #            losses[c.TOTAL_VAE] += nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1).detach()

        if classification:
            #TODO: INSERT MORE OPTIONS ON HOW TRAINING METRICS AFFECT
            ## DISJOINT VERSION

            loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu, logvar=logvar, z=z)
            loss_dict = self.loss_fn(losses, reduce_rec=True, **loss_fn_args)
            loss_dict.update(true_values=torch.tensor(-1)) # nn.BCELoss(reduction='mean')((1+mu_processed[:,:label1.size(1)])/2, label1))
            loss_dict[c.TOTAL_VAE] += -1 #nn.BCELoss(reduction='mean')((1+z[:, :label1.size(1)])/2, label1)
            losses.update({'total_vae': loss_dict['total_vae'].detach(), 'recon': loss_dict['recon'].detach(),
                           'kld': loss_dict['kld'].detach(), 'true_values': loss_dict['true_values']})

            del loss_dict

            #TODO insert MSE Classification

            err_latent = [-1] * label1.size(1)

            #TODO: insert the regression on the latents factor matching

            losses.update(prediction=nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1) )

            ## INSERT DEVICE IN THE CREATION OF EACH TENSOR
            ### AVOID COPYING FROM CPU TO GPU AS MUCH AS POSSIBLE

        return losses, {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar, "prediction": prediction,
                        'forecast': forecast, 'latents': err_latent, 'n_passed': n_passed}

    def train(self, **kwargs):

        if 'output'  in kwargs.keys():
            out_path = kwargs['output']
            track_changes=True
            self.out_path = out_path #TODO: Not happy with this thing

        else: track_changes=False;

        if track_changes:
            print("## Initializing Train indexes")
            print("->path chosen::",out_path+"/train_runs")

        ## SAVE INITIALIZATION ##
        self.save_checkpoint()

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

            for internal_iter, (x_true1, label1, y_true1, examples) in enumerate(self.data_loader):

                losses = {'total_vae':0}

                x_true1 = x_true1.to(self.device)
                label1 = label1[:, 1:].to(self.device)
                y_true1 = y_true1.to(self.device, dtype=torch.long)

                ###configuration for dsprites

                losses, params = self.vae_classification(losses, x_true1, label1, y_true1, examples,
                                                         classification=start_classification)

                self.optim_G.zero_grad()
                self.class_G.zero_grad()

                if (internal_iter%self.show_loss)==0: print("Losses:", losses)

                if not start_classification:
                    losses[c.TOTAL_VAE].backward(retain_graph=False)
                    #losses['true_values'].backward(retain_graph=False)
                    self.optim_G.step()

                if start_classification:   # and (params['n_passed']>0):
                    losses['prediction'].backward(retain_graph=False)
                    self.class_G.step()

                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum /( internal_iter+1) ## ADDED +1 HERE IDK WHY NOT BEFORE!!!!!

                ## Insert losses -- only in training set
                if track_changes and (internal_iter%2500)==0:
                    #TODO: set the tracking at a given iter_number/epoch

                    Iterations.append(internal_iter + 1)
                    Epochs.append(epoch)
                    Reconstructions.append(losses['recon'].item())
                    KLDs.append(losses['kld'].item())
                    True_Values.append(losses['true_values'].item())
                    latent_errors.append(params['latents'])

                    if not start_classification: #RECONSTRUCTION ERROR + KLD + MSE on Z
                        Accuracies, F1_scores = [-1] * len(Iterations), [-1] * len(Iterations)

                    else: #CLASSIFICATION

                        Accuracies.append(losses['prediction'].item())
                        f1_class = Accuracy_Loss()
                        F1_scores.append(f1_class(params['prediction'], y_true1).item())
                        del f1_class

                    if (internal_iter%2500)==0:
                        sofar = pd.DataFrame(data=np.array([Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores]).T,
                                             columns=['iter', 'epoch', 'reconstruction_error', 'kld', 'latent_error', 'classification_error', 'accuracy'], )
                        for i in range(label1.size(1)):
                            sofar['latent%i'%i] = np.asarray(latent_errors)[:,i]

                        sofar.to_csv(os.path.join(out_path+'/train_runs', 'metrics.csv'), index=False)
                        del sofar

#                        if not self.dataframe_eval.empty:
 #                           self.dataframe_eval.to_csv(os.path.join(out_path, 'dis_metrics.csv'), index=False)

                # TESTSET LOSSES
                if is_time_for(self.iter, self.evaluate_iter):

                    #                    self.dataframe_eval = self.dataframe_eval.append(self.evaluate_results,  ignore_index=True)
                    # test the behaviour on other losses
                    trec, tkld, tlat, tbce, tacc, I, I_tot = self.test(end_of_epoch=False)
                    factors = pd.DataFrame(
                        {'iter': self.iter, 'rec': trec, 'kld': tkld, 'latent': tlat, 'BCE': tbce, 'Acc': tacc,
                         'I': I_tot}, index=[0])

                    for i in range(len(I)):
                        factors['I_%i' % i] = np.asarray(I)[i]

                    self.dataframe_eval = self.dataframe_eval.append(factors, ignore_index=True)
                    self.net_mode(train=True)

                    if track_changes and not self.dataframe_eval.empty:
                        self.dataframe_eval.to_csv(os.path.join(out_path, 'eval_results/test_metrics.csv'),
                                                   index=False)
                        print('Saved test_metrics')

                    # include disentanglement metrics
                    dis_metrics = pd.DataFrame(self.evaluate_results, index=[0])
                    self.dataframe_dis = self.dataframe_dis.append(dis_metrics)

                    if track_changes and not self.dataframe_dis.empty:
                        self.dataframe_dis.to_csv(os.path.join(out_path, 'eval_results/dis_metrics.csv'),
                                                  index=False)
                        print('Saved dis_metrics')

                self.log_save(input_image=x_true1, recon_image=params['x_recon'], loss=losses)

            # end of epoch

        self.pbar.close()


    def test(self, end_of_epoch=True):
        self.net_mode(train=False)
        rec, kld, latent, BCE, Acc = 0, 0, 0, 0, 0
        I = np.zeros(self.z_dim)
        I_tot = 0

        N = 10**4
        l_dim = 7
        g_dim = 7

        z_array = np.zeros( shape=(self.batch_size*len(self.test_loader), l_dim))
        g_array = np.zeros( shape=(self.batch_size*len(self.test_loader), g_dim))

        for internal_iter, (x_true, label, y_true, _) in enumerate(self.test_loader):
            x_true = x_true.to(self.device)
            label = label[:,1:].to(self.device, dtype=torch.float32)
            y_true =  y_true.to(self.device, dtype=torch.long)

            mu, logvar = self.model.encode(x=x_true, )
            z = reparametrize(mu, logvar)

            mu_processed = torch.tanh(mu / 2)
            prediction, forecast = self.predict(latent=mu_processed)
            x_recon = self.model.decode(z=z,)

            z = np.asarray(nn.Sigmoid()(z).detach().cpu())
            g = np.asarray(label.detach().cpu())

            z_array[self.batch_size*internal_iter:self.batch_size*internal_iter+self.batch_size, :] = z
            g_array[self.batch_size*internal_iter:self.batch_size*internal_iter+self.batch_size, :] = g

#            I_batch , I_TOT = Interpretability(z, g)
 #           I += I_batch; I_tot += I_TOT

            rec+=(F.binary_cross_entropy(input=x_recon, target=x_true,reduction='sum').detach().item()/self.batch_size )
            kld+=(self._kld_loss_fn(mu, logvar).detach().item())

            if self.latent_loss == 'MSE':
                loss_bin = nn.MSELoss(reduction='mean')(mu_processed[:, :label.size(1)], 2 * label.to(dtype=torch.float32) - 1)
            elif self.latent_loss == 'BCE':
                loss_bin = nn.BCELoss(reduction='mean')((1+mu_processed[:, :label.size(1)])/2, label.to(dtype=torch.float32) )
            elif self.latent_loss == 'exact_MSE':
                mu_proessed = nn.Sigmoid()(mu/torch.sqrt( 1+ torch.exp(logvar)))
                loss_bin = nn.MSELoss(reduction='mean')(mu_proessed[:,:label.size(1)], label.to(dtype=torch.float32) )
            else:
                NotImplementedError('Wrong argument for latent loss.')

            latent+=(loss_bin.detach().item())
            del loss_bin

            BCE+=(nn.CrossEntropyLoss(reduction='mean')(prediction,
                                                        y_true).detach().item())


            Acc+=(Accuracy_Loss()(forecast,
                                   y_true).detach().item() )

        if end_of_epoch:
            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max),
                                    spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)

            #self.iter += 1
            #self.pbar.update(1)

        print('Done testing')

        I, I_tot = Interpretability(z_array, g_array, rel_factors=N)

        nrm = internal_iter + 1
        return rec/nrm, kld/nrm, latent/nrm, BCE/nrm, Acc/nrm, I/nrm, I_tot/nrm