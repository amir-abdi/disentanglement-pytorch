import os
from tqdm import tqdm
import logging

import torch
import torchvision.utils

from common.utils import grid2gif, dataset_samples
from common.dataset import return_data
import common.constants as c

DEBUG = False


class BaseDisentangler(object):
    def __init__(self, args):

        # Cuda
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'

        # Misc
        self.name = args.name

        # Output directory
        self.output_dir = os.path.join(args.output_dir, self.name)
        self.test_dir = os.path.join(args.test_dir, self.name)
        self.file_save = args.file_save
        self.gif_save = args.gif_save
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

        # Latent space
        self.z_dim = args.z_dim
        self.w_dim = args.w_dim
        self.num_labels = args.num_labels
        self.num_classes = args.num_classes

        # Weights
        self.nabla = args.nabla
        self.gamma = args.gamma
        self.eta = args.eta
        self.beta = args.beta
        self.kappa = args.kappa
        self.alpha = args.alpha
        self.delta = args.delta
        self.upsilion = args.upsilion

        # Solvers
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lr_G = args.lr_G
        self.lr_D = args.lr_D
        self.max_iter = int(args.max_iter)

        # Data
        self.num_channels = args.num_channels
        self.dset_dir = args.dset_dir
        self.dset_name = args.dset_name
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)
        self.image_size = args.image_size

        # Progress bar
        if not args.test:
            self.pbar = tqdm(total=self.max_iter)
        else:
            self.pbar = tqdm(total=self.data_loader.dataset.__len__() // self.batch_size)

        # logging
        self.info_cumulative = {}

        # logging
        self.iter = 0
        self.print_iter = args.print_iter
        self.float_iter = args.float_iter
        self.recon_iter = args.recon_iter
        self.traverse_iter = args.traverse_iter
        self.traverse_min = args.traverse_min
        self.traverse_max = args.traverse_max
        self.traverse_spacing = args.traverse_spacing
        self.traverse_z = args.traverse_z
        self.traverse_w = args.traverse_w
        self.white_line = None

        self.use_wandb = args.use_wandb
        if self.use_wandb:
            import wandb
            resume_wandb = True if args.wandb_resume_id is not None else False
            wandb.init(config=args, resume=resume_wandb, id=args.wandb_resume_id, project=c.WANDB_NAME)

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        self.ckpt_save_iter = args.ckpt_save_iter
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.nets = []
        self.net_dict = dict()
        self.optim_dict = dict()

        # model is the only attribute that all sub-classes should have
        self.model = None

    def log_save(self, **kwargs):
        if self.iter % self.ckpt_save_iter == 0:
            self.save_checkpoint()

        if self.iter % self.print_iter == 0:
            msg = '[{}]'.format(self.iter)
            for key, value in kwargs.items():
                if 'loss' in key:
                    msg += key + '={:.3f}'.format(value)
            self.pbar.write(msg)

        if self.iter % self.float_iter == 0:
            # average results
            for key, value in self.info_cumulative.items():
                self.info_cumulative[key] /= self.float_iter

            self.info_cumulative['iteration'] = self.iter
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
                    if key in self.info_cumulative.keys():
                        self.info_cumulative[key] += value
                    else:
                        self.info_cumulative[key] = value

        if self.iter % self.recon_iter == 0:
            self.visualize_recon(kwargs[c.INPUT_IMAGE], kwargs[c.RECON_IMAGE])

        if self.iter % self.traverse_iter == 0:
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing)

    def visualize_recon(self, input_image, recon_image, test=False):
        input_image = torchvision.utils.make_grid(input_image)
        recon_image = torchvision.utils.make_grid(recon_image)

        if self.white_line is None:
            self.white_line = torch.ones((3, input_image.size(1), 10)).to(self.device)

        samples = torch.cat([input_image, self.white_line, recon_image], dim=2)

        if self.file_save:
            if test:
                file_name = os.path.join(self.test_dir, '{}_{}.{}'.format(c.RECONSTRUCTION, self.iter, c.JPG))
            else:
                file_name = os.path.join(self.output_dir, '{}.{}'.format(c.RECONSTRUCTION, c.JPG))
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
            sample_images_dict, sample_labels_dict = dataset_samples(self.data_loader.dataset, self.device)
        else:
            sample_images, sample_labels = data
            sample_images_dict = {}
            for i, img in enumerate(sample_images):
                sample_images_dict.update({str(i): img})

        # todo: handle sample_labels_dict if fader networks got implemented

        encodings = dict()
        for key, value in sample_images_dict.items():
            encodings[key] = self.encode(value)

        gifs = []
        for key in encodings:
            latent_orig = encodings[key]
            samples = []

            # encode original on the first row
            sample = torch.sigmoid(self.decode(latent_orig.detach()))
            for _ in interp_values:
                samples.append(sample)

            if self.traverse_w:
                for lid in range(self.num_labels):
                    for lid_id in range(self.w_dim):
                        for val in interp_values:
                            latent = latent_orig.clone()
                            self.set_w(latent, lid, lid_id, val)
                            sample = torch.sigmoid(self.decode(latent)).detach()

                            samples.append(sample)
                            gifs.append(sample)

            if self.traverse_z:
                for zid in range(self.z_dim):
                    for val in interp_values:
                        latent = latent_orig.clone()
                        latent[:, zid] = val
                        self.set_z(latent, zid, val)
                        sample = torch.sigmoid(self.decode(latent))

                        samples.append(sample)
                        gifs.append(sample)

            samples = torch.cat(samples, dim=0).cpu()
            samples = torchvision.utils.make_grid(samples, nrow=num_cols)

            if self.file_save:
                if test:
                    file_name = os.path.join(self.test_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, self.iter, key, c.JPG))
                else:
                    file_name = os.path.join(self.output_dir, '{}_{}.{}'.format(c.TRAVERSE, key, c.JPG))
                torchvision.utils.save_image(samples, file_name)

            if self.use_wandb:
                import wandb
                title = '{}_{}_iter:{}'.format(c.TRAVERSE, key, self.iter)
                wandb.log({'{}_{}'.format(c.TRAVERSE, key): wandb.Image(samples, caption=title)},
                          step=self.iter)

        if self.gif_save:
            total_rows = self.num_labels * self.w_dim + self.z_dim
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(encodings), total_rows, num_cols,
                             self.num_channels, self.image_size, self.image_size).transpose(1, 2)
            for i, key in enumerate(encodings.keys()):
                for j, val in enumerate(interp_values):
                    file_name = os.path.join(self.output_dir, '{}_{}_{}.{}'.format(c.TEMP, key, j, c.JPG))
                    torchvision.utils.save_image(tensor=gifs[i][j].cpu(),
                                                 filename=file_name,
                                                 nrow=total_rows, pad_value=1)
                if test:
                    file_name = os.path.join(self.test_dir, '{}_{}_{}.{}'.format(c.GIF, self.iter, key, c.GIF))
                else:
                    file_name = os.path.join(self.output_dir, '{}_{}.{}'.format(c.GIF, key, c.GIF))

                grid2gif(str(os.path.join(self.output_dir, '{}_{}*.{}').format(c.TEMP, key, c.JPG)),
                         file_name, delay=10)

                # Delete temp image files
                for j, val in enumerate(interp_values):
                    os.remove(os.path.join(self.output_dir, '{}_{}_{}.{}'.format(c.TEMP, key, j, c.JPG)))

    def save_checkpoint(self, ckptname='last'):
        model_states = dict()
        optim_states = dict()
        for key, value in self.net_dict.items():
            if isinstance(value, dict):
                list_state_dicts = {}
                for sub_key, net in value.items():
                    # list_state_dicts.append(net.state_dict())
                    list_state_dicts.update({sub_key: net.state_dict()})
                model_states.update({key: list_state_dicts})
            else:
                model_states.update({key: value.state_dict()})

        for key, value in self.optim_dict.items():
            if isinstance(value, dict):
                list_state_dicts = {}
                for sub_key, net in value.items():
                    list_state_dicts.update({sub_key: net.state_dict()})
                optim_states.update({key: list_state_dicts})
            else:
                optim_states.update({key: value.state_dict()})

        states = {'iter': self.iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        while True:
            try:
                with open(filepath, 'wb+') as f:
                    torch.save(states, f)
                break
            except KeyboardInterrupt:
                pass

        logging.info(">>> saved checkpoint '{}' (iter {})".format(os.path.join(os.getcwd(), filepath), self.iter))

    def load_checkpoint(self, filepath, load_iternum=True):
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

            self.pbar.update(self.iter)
            logging.info("Model Loaded: {} @ iter {}".format(filepath, self.iter))

        else:
            logging.error("File does not exist: {}".format(filepath))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def encode(self, input_batch):
        if len(input_batch.size()) == 3:
            input_batch = input_batch.unsqueeze(0)
        return self.model.encode(input_batch)

    def decode(self, input_batch):
        if len(input_batch.size()) == 1:
            input_batch = input_batch.unsqueeze(0)
        return self.model.decode(input_batch)

    @staticmethod
    def set_z(z, latent_id, val):
        z[:, latent_id] = val

    def set_w(self, w, label_id, latent_id, val):
        w[:, label_id * self.w_dim + latent_id] = val
