import argparse
import numpy as np
import torch
import os
import logging
from imp import reload

from common.utils import str2bool, str2list, StyleFormatter
import models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


def main(args):
    model = getattr(models, args.alg)
    net = model(args)
    if not args.test:
        net.train()
    else:
        net.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Factor-VAE')

    # name
    parser.add_argument('--name', default='unknown_experiment', type=str, help='name of the experiment')
    parser.add_argument('--encoder_name', default='BasicEncoder64', type=str, help='name of the encoder network')
    parser.add_argument('--decoder_name', default='BasicDecoder64', type=str, help='name of the decoder network')
    parser.add_argument('--alg', type=str, help='name of the disentanglement algorithm', choices=('AE', 'VAE'))

    # cuda
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')

    parser.add_argument('--test', default=False, type=str2bool, help='to test')

    # training hyper-params
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_disc_layers', default=5, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_disc_layers', default=1000, type=int, help='size of fc layers in discriminators')

    # latent encoding
    parser.add_argument('--z_dim', default=16, type=int, help='size of the encoded z space')
    parser.add_argument('--include_labels', default=None, type=str, help='Labels (indices or names) to include in '
                                                                         'latent encoding.')
    parser.add_argument('--w_dim', default=0, type=str, help='size of the encoded w space (for each label)')
    parser.add_argument('--num_classes', default='2', type=str, help='number of classes per each label included in '
                                                                     'the label_idx flag')

    # optimizer
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 parameter of the Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 parameter of the Adam optimizer')
    parser.add_argument('--lr_G', default=1e-4, type=float, help='learning rate of the main autoencoder')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of all the discriminators')

    # Weights
    parser.add_argument('--eta', default=1.0, type=float, help='eta hyperparameter')
    parser.add_argument('--gamma', default=1.0, type=float, help='gamma hyperparameter')
    parser.add_argument('--beta', default=1.0, type=float, help='beta hyperparameter as in betaVAE')
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha hyperparameter')
    parser.add_argument('--nabla', default=0.0, type=float, help='nabla hyperparameter')
    parser.add_argument('--kappa', default=0.0, type=float, help='kappa hyperparameter')
    parser.add_argument('--delta', default=0.0, type=float, help='delta hyperparameter')
    parser.add_argument('--upsilion', default=0.0, type=float, help='upsilion hyperparameter')

    # Dataset
    parser.add_argument('--dset_dir', default='data', type=str, help='main dataset directory')
    parser.add_argument('--dset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='width and height of image')
    parser.add_argument('--num_channels', default=3, type=int, help='number of input image channels')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for the data loader')

    # Logging and visualization
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--test_dir', default='test', type=str, help='test output directory')
    parser.add_argument('--file_save', default=True, type=str2bool, help='whether to save generated images to file')
    parser.add_argument('--gif_save', default=True, type=str2bool, help='whether to save generated GIFs to file')
    parser.add_argument('--use_wandb', default=False, type=str2bool, help='use wandb for logging')
    parser.add_argument('--wandb_resume_id', default=None, type=str, help='resume previous wandb run with id')
    parser.add_argument('--traverse_spacing', default=0.5, type=float, help='spacing to traverse latents')
    parser.add_argument('--traverse_min', default=-4, type=float, help='min limit to traverse latents')
    parser.add_argument('--traverse_max', default=+4, type=float, help='max limit to traverse latents')
    parser.add_argument('--traverse_z', default=False, type=str2bool, help='whether to traverse the z space')
    parser.add_argument('--traverse_w', default=False, type=str2bool, help='whether to traverse the w space')
    parser.add_argument('--verbose', default=20, type=int, help='verbosity level')

    # Network hyper-params
    parser.add_argument('--net_size', default=1, type=int, help='Downsizing factor of the network')

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

    args = parser.parse_args()

    if args.all_iter is not None:
        args.float_iter = args.all_iter
        args.recon_iter = args.all_iter
        args.traverse_iter = args.all_iter
        args.print_iter = args.all_iter

    # Handle list arguments
    args.include_labels = str2list(args.include_labels, str)
    args.num_classes = str2list(args.num_classes, int)

    assert args.image_size == 64, 'for now, models are hard coded to support only image size of 64x63'

    # consider the same num classes for all labels (features) if a single num_classes value was provided
    args.num_labels = 0
    if args.include_labels is not None:
        args.num_labels = len(args.include_labels)
    if len(args.num_classes) == 1 and args.num_labels > 1:
        args.num_classes = args.num_classes * args.num_labels

    # makedirs
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    assert os.path.exists(args.dset_dir), 'main dataset directory does not exist'

    # verbosity
    reload(logging)  # to turn off any changes to logging done by other imported libraries
    h = logging.StreamHandler()
    h.setFormatter(StyleFormatter())
    h.setLevel(0)
    logging.root.addHandler(h)
    logging.root.setLevel(args.verbose)

    # test
    args.ckpt_load_iternum = False if args.test else args.ckpt_load_iternum
    args.use_wandb = False if args.test else args.use_wandb
    args.file_save = True if args.test else args.file_save
    args.gif_save = True if args.test else args.gif_save

    main(args)
