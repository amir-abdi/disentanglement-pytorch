import os
from tqdm import tqdm
import logging

import torch
import torchvision.utils

from common.utils import grid2gif, get_data_for_visualization, prepare_data_for_visualization, get_lr, is_time_for
from common.data_loader import _get_dataloader_with_labels
import common.constants as c
from aicrowd.aicrowd_utils import is_on_aicrowd_server, evaluate_disentanglement_metric
from common.utils import get_scheduler

DEBUG = False


class BaseDisentangler(object):
    def __init__(self, args):

        # Cuda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info('Device: {}'.format(self.device))

        # Misc
        self.name = args.name
        self.alg = args.alg
        self.controlled_capacity_increase = args.controlled_capacity_increase
        self.loss_terms = args.loss_terms
        self.on_aicrowd_server = is_on_aicrowd_server()
        self.evaluation_metric = args.evaluation_metric
        self.lr_scheduler = None
        self.w_recon_scheduler = None
        self.optim_G = None

        # Output directory
        self.train_output_dir = os.path.join(args.train_output_dir, self.name)
        self.test_output_dir = os.path.join(args.test_output_dir, self.name)
        self.file_save = args.file_save
        self.gif_save = args.gif_save
        os.makedirs(self.train_output_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Latent space
        self.z_dim = args.z_dim
        self.l_dim = args.l_dim
        self.num_labels = args.num_labels

        # Loss weights
        self.w_recon = args.w_recon

        # Solvers
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lr_G = args.lr_G
        self.lr_D = args.lr_D
        self.max_iter = int(args.max_iter)
        self.max_epoch = int(args.max_epoch)

        # Data
        self.dset_dir = args.dset_dir
        self.dset_name = args.dset_name
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.aicrowd_challenge = args.aicrowd_challenge

        from common.data_loader import get_dataloader
        self.data_loader = get_dataloader(args.dset_name, args.dset_dir, args.batch_size, args.seed, args.num_workers,
                                          args.image_size, args.include_labels, args.pin_memory, not args.test,
                                          not args.test)

        # only used if some supervision was imposed such as in Conditional VAE
        if self.data_loader.dataset.has_labels():
            self.num_classes = self.data_loader.dataset.num_classes()
            self.total_num_classes = sum(self.data_loader.dataset.num_classes(False))
            self.class_values = self.data_loader.dataset.class_values()

        self.num_channels = self.data_loader.dataset.num_channels()
        self.num_batches = len(self.data_loader)

        logging.info('Number of samples: {}'.format(len(self.data_loader.dataset)))
        logging.info('Number of batches per epoch: {}'.format(self.num_batches))
        logging.info('Number of channels: {}'.format(self.num_channels))

        # Progress bar
        if not args.test:
            self.max_iter = min(self.max_iter, self.max_epoch * self.num_batches)
            self.pbar = tqdm(total=self.max_iter)
        else:
            self.pbar = tqdm(self.num_batches)

        # logging
        self.info_cumulative = {}

        # logging
        self.iter = 0
        self.epoch = 0
        self.evaluate_results = dict()

        # logging iterations
        self.print_iter = args.print_iter if args.print_iter else self.num_batches
        self.float_iter = args.float_iter if args.float_iter else self.num_batches
        self.recon_iter = args.recon_iter if args.recon_iter else self.num_batches
        self.traverse_iter = args.traverse_iter if args.traverse_iter else self.num_batches
        self.evaluate_iter = args.evaluate_iter if args.evaluate_iter else self.num_batches
        self.ckpt_save_iter = args.ckpt_save_iter if args.ckpt_save_iter else self.num_batches
        self.schedulers_iter = args.schedulers_iter if args.schedulers_iter else self.num_batches

        # override logging iterations if all_iter is set (except for the schedulers_iter)
        if args.all_iter:
            self.float_iter = args.all_iter
            self.recon_iter = args.all_iter
            self.traverse_iter = args.all_iter
            self.print_iter = args.all_iter
            self.evaluate_iter = args.all_iter
            self.ckpt_save_iter = args.all_iter
            self.schedulers_iter = args.all_iter

        if args.treat_iter_as_epoch:
            self.float_iter = self.float_iter * self.num_batches
            self.recon_iter = self.recon_iter * self.num_batches
            self.traverse_iter = self.traverse_iter * self.num_batches
            self.print_iter = self.print_iter * self.num_batches
            self.evaluate_iter = self.evaluate_iter * self.num_batches
            self.ckpt_save_iter = self.ckpt_save_iter * self.num_batches
            self.schedulers_iter = self.schedulers_iter * self.num_batches

        # traversing the latent space
        self.traverse_min = args.traverse_min
        self.traverse_max = args.traverse_max
        self.traverse_spacing = args.traverse_spacing
        self.traverse_z = args.traverse_z
        self.traverse_l = args.traverse_l
        self.traverse_c = args.traverse_c
        self.white_line = None

        self.use_wandb = args.use_wandb
        if self.use_wandb:
            import wandb
            resume_wandb = True if args.wandb_resume_id else False
            wandb.init(config=args, resume=resume_wandb, id=args.wandb_resume_id, project=c.WANDB_NAME,
                       name=args.name)

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.net_dict = dict()
        self.optim_dict = dict()

        # model is the only attribute that all sub-classes should have
        self.model = None

        # FactorVAE args
        self.ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device, requires_grad=False)
        self.zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device, requires_grad=False)
        self.num_layer_disc = args.num_layer_disc
        self.size_layer_disc = args.size_layer_disc

        # FactorVAE & BetaTCVAE args
        self.w_tc = args.w_tc

        # InfoVAE args
        self.w_infovae = args.w_infovae

        # DIPVAE args
        self.w_dipvae = args.w_dipvae

        # DIPVAE args
        self.lambda_od = args.lambda_od
        self.lambda_d_factor = args.lambda_d_factor
        self.lambda_d = self.lambda_d_factor * self.lambda_od

    def log_save(self, **kwargs):
        self.step()

        # don't log anything if running on the aicrowd_server
        if self.on_aicrowd_server:
            return

        # save a checkpoint every ckpt_save_iter
        if is_time_for(self.iter, self.ckpt_save_iter):
            self.save_checkpoint()

        if is_time_for(self.iter, self.print_iter):
            msg = '[{}:{}]  '.format(self.epoch, self.iter)
            for key, value in kwargs.get(c.LOSS, dict()).items():
                msg += '{}_{}={:.3f}  '.format(c.LOSS, key, value)
            for key, value in kwargs.get(c.ACCURACY, dict()).items():
                msg += '{}_{}={:.3f}  '.format(c.ACCURACY, key, value)
            self.pbar.write(msg)

        # visualize the reconstruction of the current batch every recon_iter
        if is_time_for(self.iter, self.recon_iter):
            self.visualize_recon(kwargs[c.INPUT_IMAGE], kwargs[c.RECON_IMAGE])

        # traverse the latent factors every traverse_iter
        if is_time_for(self.iter, self.traverse_iter):
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing)

        # if any evaluation is included in args.evaluate_metric, evaluate every evaluate_iter
        if self.evaluation_metric and is_time_for(self.iter, self.evaluate_iter):
            self.evaluate_results = evaluate_disentanglement_metric(self, metric_names=self.evaluation_metric)

        # log scalar values using wandb
        if is_time_for(self.iter, self.float_iter):
            # average results
            for key, value in self.info_cumulative.items():
                self.info_cumulative[key] /= self.float_iter

            # other values to log
            self.info_cumulative[c.ITERATION] = self.iter
            self.info_cumulative[c.LEARNING_RATE] = get_lr(self.optim_dict['optim_G'])  # assuming we want optim_G

            # todo: not happy with this architecture for logging... should make it easier to add new variables to log
            if self.evaluation_metric:
                for key, value in self.evaluate_results.items():
                    self.info_cumulative[key] = value

            if self.use_wandb:
                import wandb
                wandb.log(self.info_cumulative, step=self.iter)

            # empty info_cumulative
            for key, value in self.info_cumulative.items():
                self.info_cumulative[key] = 0
        else:
            # accumulate results
            for key, value in kwargs.items():
                if isinstance(value, float):
                    self.info_cumulative[key] = value + self.info_cumulative.get(key, 0)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        complex_key = key + '/' + subkey
                        self.info_cumulative[complex_key] = float(subvalue) + self.info_cumulative.get(complex_key, 0)

        # update schedulers
        if is_time_for(self.iter, self.schedulers_iter):
            self.schedulers_step(kwargs.get(c.LOSS, dict()).get(c.TOTAL_VAE_EPOCH, 0),
                                 self.iter // self.schedulers_iter)

    def visualize_recon(self, input_image, recon_image, test=False):
        input_image = torchvision.utils.make_grid(input_image)
        recon_image = torchvision.utils.make_grid(recon_image)

        if self.white_line is None:
            self.white_line = torch.ones((3, input_image.size(1), 10)).to(self.device)

        samples = torch.cat([input_image, self.white_line, recon_image], dim=2)

        if self.file_save:
            if test:
                file_name = os.path.join(self.test_output_dir, '{}_{}.{}'.format(c.RECON, self.iter, c.JPG))
            else:
                file_name = os.path.join(self.train_output_dir, '{}.{}'.format(c.RECON, c.JPG))
            torchvision.utils.save_image(samples, file_name)

        if self.use_wandb:
            import wandb
            wandb.log({c.RECON_IMAGE: wandb.Image(samples, caption=str(self.iter))},
                      step=self.iter)

    def visualize_traverse(self, limit: tuple, spacing, data=None, test=False):
        self.net_mode(train=False)
        interp_values = torch.arange(limit[0], limit[1], spacing)
        num_cols = interp_values.size(0)

        if data is None:
            sample_images_dict, sample_labels_dict = get_data_for_visualization(self.data_loader.dataset, self.device)
        else:
            sample_images_dict, sample_labels_dict = prepare_data_for_visualization(data)

        encodings = dict()
        for key in sample_images_dict.keys():
            encodings[key] = self.encode_deterministic(images=sample_images_dict[key], labels=sample_labels_dict[key])

        gifs = []
        for key in encodings:
            latent_orig = encodings[key]
            label_orig = sample_labels_dict[key]
            logging.debug('latent_orig: {}, label_orig: {}'.format(latent_orig, label_orig))
            samples = []

            # encode original on the first row
            sample = self.decode(latent=latent_orig.detach(), labels=label_orig)
            for _ in interp_values:
                samples.append(sample)

            if self.traverse_l:
                for lid in range(self.num_labels):
                    for lid_id in range(self.l_dim):
                        for val in interp_values:
                            latent = latent_orig.clone()
                            self.set_l(latent, lid, lid_id, val)
                            sample = self.decode(latent=latent, labels=label_orig).detach()

                            samples.append(sample)
                            gifs.append(sample)

            if self.traverse_z:
                for zid in range(self.z_dim):
                    for val in interp_values:
                        latent = latent_orig.clone()
                        latent[:, zid] = val
                        self.set_z(latent, zid, val)
                        sample = self.decode(latent=latent, labels=label_orig)

                        samples.append(sample)
                        gifs.append(sample)

            if self.traverse_c:
                num_classes = self.data_loader.dataset.num_classes(False)
                for lid in range(self.num_labels):
                    for temp_i in range(num_cols):
                        class_id = temp_i % num_classes[lid]
                        class_value = self.class_values[lid][class_id]
                        label = label_orig.clone()
                        latent = latent_orig.clone()
                        new_label = torch.tensor(class_value).to(self.device, dtype=torch.long).unsqueeze(0)
                        logging.debug('label: {} --> {}'.format(label[:, lid], new_label))
                        label[:, lid] = new_label

                        # c is being traversed, so, latent should not contain the condition
                        if latent.size(1) == self.z_dim + self.total_num_classes:
                            latent = latent[:, :self.z_dim]
                        sample = self.decode(latent=latent, labels=label).detach()

                        samples.append(sample)
                        gifs.append(sample)

            samples = torch.cat(samples, dim=0).cpu()
            samples = torchvision.utils.make_grid(samples, nrow=num_cols)

            if self.file_save:
                if test:
                    file_name = os.path.join(self.test_output_dir,
                                             '{}_{}_{}.{}'.format(c.TRAVERSE, self.iter, key, c.JPG))
                else:
                    file_name = os.path.join(self.train_output_dir, '{}_{}.{}'.format(c.TRAVERSE, key, c.JPG))
                torchvision.utils.save_image(samples, file_name)

            if self.use_wandb:
                import wandb
                title = '{}_{}_iter:{}'.format(c.TRAVERSE, key, self.iter)
                wandb.log({'{}_{}'.format(c.TRAVERSE, key): wandb.Image(samples, caption=title)},
                          step=self.iter)

        if self.gif_save and len(gifs) > 0:
            total_rows = self.num_labels * self.l_dim + \
                         self.z_dim * int(self.traverse_z) + \
                         self.num_labels * int(self.traverse_c)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(encodings), total_rows, num_cols,
                             self.num_channels, self.image_size, self.image_size).transpose(1, 2)
            for i, key in enumerate(encodings.keys()):
                for j, val in enumerate(interp_values):
                    file_name = \
                        os.path.join(self.train_output_dir, '{}_{}_{}.{}'.format(c.TEMP, key, str(j).zfill(2), c.JPG))
                    torchvision.utils.save_image(tensor=gifs[i][j].cpu(),
                                                 filename=file_name,
                                                 nrow=total_rows, pad_value=1)
                if test:
                    file_name = os.path.join(self.test_output_dir, '{}_{}_{}.{}'.format(c.GIF, self.iter, key, c.GIF))
                else:
                    file_name = os.path.join(self.train_output_dir, '{}_{}.{}'.format(c.GIF, key, c.GIF))

                grid2gif(str(os.path.join(self.train_output_dir, '{}_{}*.{}').format(c.TEMP, key, c.JPG)),
                         file_name, delay=10)

                # Delete temp image files
                for j, val in enumerate(interp_values):
                    os.remove(
                        os.path.join(self.train_output_dir, '{}_{}_{}.{}'.format(c.TEMP, key, str(j).zfill(2), c.JPG)))

    def save_checkpoint(self, ckptname='last'):
        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        model_states = dict()
        optim_states = dict()

        # neural models
        for key, value in self.net_dict.items():
            if isinstance(value, dict):
                list_state_dicts = {}
                for sub_key, net in value.items():
                    list_state_dicts.update({sub_key: net.state_dict()})
                model_states.update({key: list_state_dicts})
            else:
                model_states.update({key: value.state_dict()})

        # optimizers' states
        for key, value in self.optim_dict.items():
            if isinstance(value, dict):
                list_state_dicts = {}
                for sub_key, net in value.items():
                    list_state_dicts.update({sub_key: net.state_dict()})
                optim_states.update({key: list_state_dicts})
            else:
                optim_states.update({key: value.state_dict()})

        # wrap up everything in a dict
        states = {'iter': self.iter + 1,  # to avoid saving right after loading
                  'model_states': model_states,
                  'optim_states': optim_states}

        # make sure KeyboardInterrupt exceptions don't mess up the model saving process
        while True:
            try:
                with open(filepath, 'wb+') as f:
                    torch.save(states, f)
                break
            except KeyboardInterrupt:
                pass

        logging.info("saved checkpoint '{}' @ iter:{}".format(os.path.join(os.getcwd(), filepath), self.iter))

    def load_checkpoint(self, filepath, load_iternum=True, ignore_failure=True, load_optim=True):
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            if load_iternum:
                self.iter = checkpoint['iter']

            for key, value in self.net_dict.items():
                try:
                    if isinstance(value, dict):
                        state_dicts = checkpoint['model_states'][key]
                        for sub_key, net in value.items():
                            value[sub_key].load_state_dict(state_dicts[sub_key], strict=False)
                    else:
                        value.load_state_dict(checkpoint['model_states'][key], strict=False)
                except Exception as e:
                    logging.warning("Could not load {}".format(key))
                    logging.warning(str(e))
                    if not ignore_failure:
                        raise e
            if load_optim:
                for key, value in self.optim_dict.items():
                    try:
                        if isinstance(value, dict):
                            state_dicts = checkpoint['optim_states'][key]
                            for sub_key, net in value.items():
                                value[sub_key].load_state_dict(state_dicts[sub_key])
                        else:
                            value.load_state_dict(checkpoint['optim_states'][key])
                    except Exception as e:
                        logging.warning("Could not load {}".format(key))
                        logging.warning(str(e))
                        if not ignore_failure:
                            raise e

            self.pbar.update(self.iter)
            # TODO: remove the following hard coded lr assumption on optim_G
            # Assuming optim_G to be the optimizer for the generator and the one we are interested in
            logging.info("Model Loaded: {} @ iter:{}, lr:{:.6f}".format(filepath, self.iter,
                                                                        get_lr(self.optim_dict['optim_G'])))

        else:
            logging.error("File does not exist: {}".format(filepath))

    def net_mode(self, train):
        for net in self.net_dict.values():
            if train:
                net.train()
            else:
                net.eval()

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if len(images.size()) == 3:
            images = images.unsqueeze(0)
        return self.model.encode(images)

    def encode_stochastic(self, **kwargs):
        raise NotImplementedError

    def decode(self, **kwargs):
        latent = kwargs['latent']
        if len(latent.size()) == 1:
            latent = latent.unsqueeze(0)
        return self.model.decode(latent)

    @staticmethod
    def set_z(z, latent_id, val):
        z[:, latent_id] = val

    def set_l(self, l, label_id, latent_id, val):
        l[:, label_id * self.l_dim + latent_id] = val

    def loss_fn(self, **kwargs):
        raise NotImplementedError

    def step(self):
        self.iter += 1
        self.pbar.update(1)
        self.epoch = self.iter // self.num_batches

        if self.aicrowd_challenge and self.on_aicrowd_server:
            from aicrowd import aicrowd_helpers
            aicrowd_helpers.register_progress(self.iter / self.max_iter)

    def training_complete(self):
        if self.epoch > self.max_epoch or self.iter > self.max_iter:
            logging.info("-------Training Finished----------")
            return True
        return False

    def schedulers_step(self, validation_loss=None, step_num=None):
        self.lr_scheduler_step(validation_loss)
        self.w_recon_scheduler_step(step_num)

    def lr_scheduler_step(self, validation_loss):
        if self.lr_scheduler is None:
            return
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(validation_loss)
        else:
            self.lr_scheduler.step()

    def w_recon_scheduler_step(self, step_num):
        if self.w_recon_scheduler is None:
            return
        self.w_recon = self.w_recon_scheduler.step(step_num)

    def setup_schedulers(self, lr_scheduler, lr_scheduler_args, w_recon_scheduler, w_recon_scheduler_args):
        self.lr_scheduler = get_scheduler(self.optim_G, lr_scheduler, lr_scheduler_args)
        self.w_recon_scheduler = get_scheduler(self.w_recon, w_recon_scheduler, w_recon_scheduler_args)
