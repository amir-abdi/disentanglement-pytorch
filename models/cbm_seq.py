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


class CBM_Seq(VAE):
    """
    Graybox version of VAE, with standard implementation. The discussion on
    """

    def __init__(self, args):

        super().__init__(args)

        print('Initialized CBM_Join model')

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
        #self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))
        #self.optim_G_mse = optim.Adam(self.model.encoder.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # update nets
        self.net_dict['G'] = self.model
        self.optim_dict['optim_G'] = self.optim_G

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

        ## add binary classification layer
        self.classification = nn.Linear(self.z_dim, 2, bias=False).to(self.device) ### CHANGED OUT DIMENSION
        self.reduce_rec = args.reduce_recon

        self.class_G_all = optim.Adam([*self.model.encoder.parameters(), *self.classification.parameters()],
                                      lr=self.lr_G, betas=(self.beta1, self.beta2))

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

    def loss_fn(self, input_losses, reduce_rec=False, **kwargs):
        output_losses = dict()
        output_losses['total'] = input_losses.get('total', 0)
        return output_losses

    def cbm_classification(self, losses, x_true1, label1, y_true1, examples, classification=False):

        # label_1 \in [0,1]
        mu, _ = self.model.encode(x=x_true1,)

        z = torch.tanh(mu/2)
        prediction, forecast = self.predict(latent=z)
        #x_recon = self.model.decode(z=z,)

        rn_mask = (examples == 1)

        if examples[rn_mask] is None:
            n_passed = 0
        else:
            n_passed = len(examples[rn_mask])

        loss_dict = self.loss_fn(losses, reduce_rec=False,)

        if classification:
            loss_true_vals = torch.tensor(-1)
            losses.update({'total_vae': loss_dict['total_vae'].detach(), 'true_values': -1})
            err_latent = [-1] * label1.size(1)


        else:

            losses.update(loss_dict)

            losses.update(prediction=nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1.to(self.device, dtype=torch.long)) )
            losses[c.TOTAL_VAE] += nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1.to(self.device, dtype=torch.long))

            if n_passed > 0: # added the presence of only small labelled generative factors

                ## loss of continuous variables
                if self.latent_loss == 'MSE':

                    # z \in [-1,1]
                    loss_bin = nn.MSELoss(reduction='mean')( z[rn_mask][:, :label1.size(1)], 2*label1[rn_mask]-1  )

                    ## track losses
                    err_latent = []
                    for i in range(label1.size(1)):
                        err_latent.append(nn.MSELoss(reduction='mean')(z[rn_mask][:, i], 2 * label1[rn_mask][:,i] - 1).detach().item() )

                    losses.update(true_values= loss_bin)
                    losses[c.TOTAL_VAE] += loss_bin

                elif self.latent_loss == 'BCE':

                    # z \in [-1,1]
                    loss_bin = nn.BCELoss(reduction='mean')((1+z[rn_mask][:, :label1.size(1)])/2,
                                                             label1[rn_mask] )
                    ## track losses
                    err_latent = []
                    for i in range(label1.size(1)):
                        err_latent.append(nn.BCELoss(reduction='mean')((1+z[rn_mask][:, i])/2,
                                                                        label1[rn_mask][:, i] ).detach().item())
                    losses.update(true_values= loss_bin)
                    losses[c.TOTAL_VAE] += loss_bin

                else:
                    raise NotImplementedError('Not implemented loss.')

            else:
                err_latent = [-1]*label1.size(1)
                losses.update(true_values=torch.tensor(-1))

        del loss_dict

        return losses, {'mu': mu, 'z': z, "prediction": prediction, 'forecast': forecast,
                    'latents': err_latent, 'n_passed': n_passed}

    def train(self, **kwargs):

        out_path = None

        if 'output'  in kwargs.keys():
            out_path = kwargs['output']
            track_changes=True
            self.out_path = out_path #TODO: Not happy with this thing

        else: track_changes=False;

        if track_changes:
            print("## Initializing Train indexes")
            print("::path chosen ->",out_path+"/train_runs")

        Iterations, Epochs, True_Values, Accuracies, CE_class = [], [], [], [], []  ## JUST HERE FOR NOW
        latent_errors = []
        epoch = 0
        while not self.training_complete():
            epoch += 1
            self.net_mode(train=True)
            vae_loss_sum = 0
            # add the classification layer #

            for internal_iter, (x_true1, label1, y_true1, examples) in enumerate(self.data_loader):

                if internal_iter > 1 and is_time_for(self.iter, self.evaluate_iter):
                    # test the behaviour on other losses
                    trec, tkld, tlat, tbce, tacc, I, I_tot = self.test(end_of_epoch=False)
                    factors = pd.DataFrame(
                        {'iter': self.iter, 'rec': trec, 'kld': tkld, 'latent': tlat, 'BCE': tbce, 'Acc': tacc,
                         'I': I_tot}, index=[0])

                    for i in range(label1.size(1)):
                        factors['I_%i' % i] = np.asarray(I)[i]

                    self.dataframe_eval = self.dataframe_eval.append(factors, ignore_index=True)
                    self.net_mode(train=True)

                    if track_changes and not self.dataframe_eval.empty:
                        self.dataframe_eval.to_csv(os.path.join(out_path, 'eval_results/test_metrics.csv'), index=False)
                        print('Saved test_metrics')

                    # include disentanglement metrics
                    dis_metrics = pd.DataFrame(self.evaluate_results, index=[0])
                    self.dataframe_dis = self.dataframe_dis.append(dis_metrics)

                    if track_changes and not self.dataframe_dis.empty:
                        self.dataframe_dis.to_csv(os.path.join(out_path, 'eval_results/dis_metrics.csv'), index=False)
                        print('Saved dis_metrics')

                Iterations.append(internal_iter+1)
                Epochs.append(epoch)
                losses = {'total_vae':0}

                x_true1 = x_true1.to(self.device)
                label1 = label1[:, 1:].to(self.device)
                y_true1 = y_true1.to(self.device)

                ###configuration for dsprites

                losses, params = self.cbm_classification(losses, x_true1, label1, y_true1, examples)

                self.class_G_all.zero_grad()

                if (internal_iter%self.show_loss)==0: print("Losses:", losses)

                losses[c.TOTAL_VAE].backward(retain_graph=False)
                self.class_G_all.step()

                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum /( internal_iter+1)

                ## Insert losses -- only in training set
                if track_changes:
                    #TODO: set the tracking at a given iter_number/epoch

                    True_Values.append(losses['true_values'].item())
                    latent_errors.append(params['latents'])
                    CE_class.append(losses['prediction'].item())
                    f1_class = Accuracy_Loss().to(self.device)
                    Accuracies.append(f1_class(params['forecast'], y_true1).item())

                    del f1_class

                    if (internal_iter%500)==0:
                        sofar = pd.DataFrame(data=np.array([Iterations, Epochs,  True_Values, CE_class, Accuracies]).T,
                                             columns=['iter', 'epoch', 'latent_error', 'classification_error', 'accuracy'], )
                        for i in range(label1.size(1)):
                            sofar['latent%i'%i] = np.asarray(latent_errors)[:,i]

                        sofar.to_csv(os.path.join(out_path+'/train_runs', 'metrics.csv'), index=False)
                        del sofar

                self.log_save(input_image=x_true1, recon_image=x_true1, loss=losses)
            # end of epoch

        self.pbar.close()

    def test(self, end_of_epoch=True):
        self.net_mode(train=False)
        rec, kld, latent, BCE, Acc = 0, 0, 0, 0, 0
        I = np.zeros(self.z_dim)
        I_tot = 0

        N = 10 ** 4
        l_dim = 7
        g_dim = 7

        z_array = np.zeros(shape=(self.batch_size * len(self.test_loader), l_dim))
        g_array = np.zeros(shape=(self.batch_size * len(self.test_loader), g_dim))

        for internal_iter, (x_true, label, y_true, _) in enumerate(self.test_loader):
            x_true = x_true.to(self.device)
            label = label[:, 1:].to(self.device, dtype=torch.float32)
            y_true = y_true.to(self.device, dtype=torch.long)

            mu, logvar = self.model.encode(x=x_true, )
            z = reparametrize(mu, logvar)

            mu_processed = torch.tanh(mu / 2)
            prediction, forecast = self.predict(latent=mu_processed)
            x_recon = self.model.decode(z=z, )

            z = np.asarray(nn.Sigmoid()(z).detach().cpu())
            g = np.asarray(label.detach().cpu())

            z_array[self.batch_size * internal_iter:self.batch_size * internal_iter + self.batch_size, :] = z
            g_array[self.batch_size * internal_iter:self.batch_size * internal_iter + self.batch_size, :] = g

            #            I_batch , I_TOT = Interpretability(z, g)
            #           I += I_batch; I_tot += I_TOT

            rec += (F.binary_cross_entropy(input=x_recon, target=x_true,
                                           reduction='sum').detach().item() / self.batch_size)
            kld += (self._kld_loss_fn(mu, logvar).detach().item())

            if self.latent_loss == 'MSE':
                loss_bin = nn.MSELoss(reduction='mean')(mu_processed[:, :label.size(1)],
                                                        2 * label.to(dtype=torch.float32) - 1)
            elif self.latent_loss == 'BCE':
                loss_bin = nn.BCELoss(reduction='mean')((1 + mu_processed[:, :label.size(1)]) / 2,
                                                        label.to(dtype=torch.float32))
            elif self.latent_loss == 'exact_MSE':
                mu_proessed = nn.Sigmoid()(mu / torch.sqrt(1 + torch.exp(logvar)))
                loss_bin = nn.MSELoss(reduction='mean')(mu_proessed[:, :label.size(1)],
                                                        label.to(dtype=torch.float32))
            else:
                NotImplementedError('Wrong argument for latent loss.')

            latent += (loss_bin.detach().item())
            del loss_bin

            BCE += (nn.CrossEntropyLoss(reduction='mean')(prediction,
                                                          y_true).detach().item())

            Acc += (Accuracy_Loss()(forecast,
                                    y_true).detach().item())

        if end_of_epoch:
            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max),
                                    spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)

            # self.iter += 1
            # self.pbar.update(1)

        print('Done testing')

        I, I_tot = Interpretability(z_array, g_array, rel_factors=N)

        nrm = internal_iter + 1
        return rec / nrm, kld / nrm, latent / nrm, BCE / nrm, Acc / nrm, I / nrm, I_tot / nrm