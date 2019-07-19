import os
from tqdm import tqdm
import logging

import torch
import torchvision.utils

from common.utils import grid2gif, get_data_for_visualization, prepare_data_for_visualization
from common.dataset import get_dataloader
import common.constants as c

DEBUG = False


class BaseDisentangler(object):
    def __init__(self, args):

        # Cuda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info('Device: {}'.format(self.device))

        # Misc
        self.name = args.name
        self.alg = args.alg
        self.vae_loss = args.vae_loss
        self.vae_type = args.vae_type

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

        # Data
        self.dset_dir = args.dset_dir
        self.dset_name = args.dset_name
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        if args.aicrowd_challenge:
            import utils_pytorch as aicrowd
            kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if self.device == 'cuda' else {}
            self.data_loader = aicrowd.get_loader(batch_size=args.batch_size, **kwargs)
        else:
            self.data_loader = get_dataloader(args)

            # only used if some supervision was imposed such as in Conditional VAE
            self.num_classes = self.data_loader.dataset.num_classes()
            self.total_num_classes = sum(self.data_loader.dataset.num_classes(False))
            self.class_values = self.data_loader.dataset.class_values()

        self.num_channels = self.data_loader.dataset.num_channels()
        # self.num_channels = self.data_loader.dataset.observation_shape[2]

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
        self.traverse_l = args.traverse_l
        self.traverse_c = args.traverse_c
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

        self.net_dict = dict()
        self.optim_dict = dict()

        # model is the only attribute that all sub-classes should have
        self.model = None

    def log_save(self, **kwargs):
        if self.iter > 0 and self.iter % self.ckpt_save_iter == 0:
            self.save_checkpoint()

        if self.iter % self.print_iter == 0:
            msg = '[{}]  '.format(self.iter)
            for key, value in kwargs.get(c.LOSS, dict()).items():
                msg += '{}_{}={:.3f}  '.format(c.LOSS, key, value)
            for key, value in kwargs.get(c.ACCURACY, dict()).items():
                msg += '{}_{}={:.3f}  '.format(c.ACCURACY, key, value)
            self.pbar.write(msg)

        if self.iter % self.float_iter == 0:
            # average results
            for key, value in self.info_cumulative.items():
                self.info_cumulative[key] /= self.float_iter

            self.info_cumulative[c.ITERATION] = self.iter
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
                        complex_key = key + '_' + subkey
                        self.info_cumulative[complex_key] = float(subvalue) + self.info_cumulative.get(complex_key, 0)

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
            encodings[key] = self.encode_deterministic(images=sample_images_dict[key],
                                                       labels=sample_labels_dict[key])

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
                    file_name = os.path.join(self.test_output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, self.iter, key, c.JPG))
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

        states = {'iter': self.iter + 1,  # to avoid saving right after loading
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

        logging.info("saved checkpoint '{}' @ iter:{}".format(os.path.join(os.getcwd(), filepath), self.iter))

    def load_checkpoint(self, filepath, load_iternum=True, ignore_failure=True):
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
            logging.info("Model Loaded: {} @ iter:{}".format(filepath, self.iter))

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
