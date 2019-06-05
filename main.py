'''
TODO: add author and license info to all files.
TODO: 3 different divergences in the InfoVAE paper https://arxiv.org/pdf/1706.02262.pdf
'''

import sys
import argparse
import numpy as np
import torch
import os
import logging
from imp import reload

from common.utils import str2bool, str2list, StyleFormatter
from common import constants as c
import models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


def main(args):
    model_cl = getattr(models, args.alg)
    model = model_cl(args)
    if args.ckpt_load:
        model.load_checkpoint(args.ckpt_load, load_iternum=args.ckpt_load_iternum)

    if not args.test:
        model.train()
    else:
        model.test()


def get_args(sys_args):
    parser = argparse.ArgumentParser(description='disentanglement-pytorch')

    # name
    parser.add_argument('--alg', type=str,
                        help='the disentanglement algorithm',
                        choices=c.ALGS)
    parser.add_argument('--vae_loss', type=str,
                        help='type of VAE loss',
                        choices=c.VAE_LOSS)
    parser.add_argument('--name', default='unknown_experiment', type=str,
                        help='name of the experiment')
    parser.add_argument('--encoder', default='SimpleGaussianEncoder64', type=str,
                        help='name of the encoder network',
                        choices=('SimpleEncoder64', 'SimpleGaussianEncoder64',))
    parser.add_argument('--decoder', default='SimpleDecoder64', type=str,
                        help='name of the decoder network',
                        choices=('SimpleDecoder64',))
    parser.add_argument('--discriminator', default='SimpleDiscriminator', type=str,
                        help='the discriminator network',
                        choices=('SimpleDiscriminator',))
    parser.add_argument('--label_tiler', default='MultiTo2DChannel', type=str,
                        help='the tile network used to convert one hot labels to 2D channels',
                        choices=('MultiTo2DChannel',)
                        )

    # Test or train
    parser.add_argument('--test', default=False, type=str2bool, help='to test')

    # training hyper-params
    parser.add_argument('--max_iter', default=3e7, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_disc_layers', default=5, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_disc_layers', default=1000, type=int, help='size of fc layers in discriminators')

    # latent encoding
    parser.add_argument('--z_dim', default=16, type=int, help='size of the encoded z space')
    parser.add_argument('--include_labels', default=None, type=str, help='Labels (indices or names) to include in '
                                                                         'latent encoding.')
    parser.add_argument('--l_dim', default=0, type=str, help='size of the encoded w space (for each label)')

    # optimizer
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 parameter of the Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 parameter of the Adam optimizer')
    parser.add_argument('--lr_G', default=1e-6, type=float, help='learning rate of the main autoencoder')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of all the discriminators')

    # Architecture hyper-parameters
    parser.add_argument('--num_layer_disc', default=6, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_layer_disc', default=1000, type=int, help='size of fc layers in discriminators')

    # Loss weights and parameters [Common]
    # parser.add_argument('--mult_all_w', default=100.0, type=float, help='multiply all weights with this value')
    parser.add_argument('--w_recon', default=1.0, type=float, help='reconstruction loss weight')
    parser.add_argument('--w_kld', default=1.0, type=float, help='main KLD loss weight (e.g. in BetaVAE)')

    # Loss weights and parameters [CapacityVAE]
    parser.add_argument('--max_c', default=25.0, type=float, help='maximum value of control parameter in CapacityVAE')
    parser.add_argument('--iterations_c', default=1000000, type=int, help='how many iterations to reach max_c')

    # Loss weights and parameters [CapacityVAE]
    parser.add_argument('--w_tc', default=1.0, type=float, help='total correlation loss weight (e.g. in BetaVAE)')

    # Dataset
    parser.add_argument('--dset_dir', default='data', type=str, help='main dataset directory')
    parser.add_argument('--dset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='width and height of image')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for the data loader')

    # Logging and visualization
    parser.add_argument('--train_output_dir', default='train_outputs', type=str, help='output directory')
    parser.add_argument('--test_output_dir', default='test_outputs', type=str, help='test output directory')
    parser.add_argument('--file_save', default=True, type=str2bool, help='whether to save generated images to file')
    parser.add_argument('--gif_save', default=True, type=str2bool, help='whether to save generated GIFs to file')
    parser.add_argument('--use_wandb', default=False, type=str2bool, help='use wandb for logging')
    parser.add_argument('--wandb_resume_id', default=None, type=str, help='resume previous wandb run with id')
    parser.add_argument('--traverse_spacing', default=0.5, type=float, help='spacing to traverse latents')
    parser.add_argument('--traverse_min', default=-4, type=float, help='min limit to traverse latents')
    parser.add_argument('--traverse_max', default=+4, type=float, help='max limit to traverse latents')
    parser.add_argument('--traverse_z', default=False, type=str2bool, help='whether to traverse the z space')
    parser.add_argument('--traverse_l', default=False, type=str2bool, help='whether to traverse the l space')
    parser.add_argument('--traverse_c', default=False, type=str2bool, help='whether to traverse the condition')
    parser.add_argument('--verbose', default=20, type=int, help='verbosity level')

    # Save/Load checkpoint
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_load_iternum', default=True, type=str2bool, help='start global iteration from ckpt')
    parser.add_argument('--ckpt_save_iter', default=2000, type=int, help='checkpoint save iter')

    # Iterations
    parser.add_argument('--float_iter', default=100, type=int, help='number of iterations to aggregate float logs')
    parser.add_argument('--recon_iter', default=2000, type=int, help='iterations to reconstruct input image')
    parser.add_argument('--traverse_iter', default=2000, type=int, help='iterations to traverse latent spaces')
    parser.add_argument('--print_iter', default=500, type=int, help='iterations to print float values')
    parser.add_argument('--all_iter', default=None, type=int, help='use same iteration for all')

    args = parser.parse_args(sys_args)

    if args.all_iter is not None:
        args.float_iter = args.all_iter
        args.recon_iter = args.all_iter
        args.traverse_iter = args.all_iter
        args.print_iter = args.all_iter

    # Handle list arguments
    args.include_labels = str2list(args.include_labels, str)

    assert args.image_size == 64, 'for now, models are hard coded to support only image size of 64x63'

    args.num_labels = 0
    if args.include_labels is not None:
        args.num_labels = len(args.include_labels)

    # makedirs
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)
    assert os.path.exists(args.dset_dir), 'main dataset directory does not exist'

    # test
    args.ckpt_load_iternum = False if args.test else args.ckpt_load_iternum
    args.use_wandb = False if args.test else args.use_wandb
    args.file_save = True if args.test else args.file_save
    args.gif_save = True if args.test else args.gif_save

    return args


if __name__ == "__main__":
    args_ = get_args(sys.argv[1:])

    # verbosity
    reload(logging)  # to turn off any changes to logging done by other imported libraries
    h = logging.StreamHandler()
    h.setFormatter(StyleFormatter())
    h.setLevel(0)
    logging.root.addHandler(h)
    logging.root.setLevel(args_.verbose)

    main(args_)
