'''
TODO: add author and license info to all files.
TODO: infoVAE
TODO: 3 different divergences in the InfoVAE paper https://arxiv.org/pdf/1706.02262.pdf
TODO: evaluation metrics
TODO: Add Adversarial Autoencoders https://arxiv.org/pdf/1511.05644.pdf
TODO: A version of CVAE where independence between C and Z is enforced
TODO: Add PixelCNN and PixelCNN++ and PixelVAE
TODO: Add VQ-VAE (discrete encodings) and VQ-VAE2 --> I guess Version 2 has pixelCNN
TODO: SCGAN_Disentangled_Representation_Learning_by_Addi

TODO: Update license to MIT
'''

import sys
import argparse
import numpy as np
import torch
import os
import logging
from importlib import reload

from common.utils import str2bool, StyleFormatter, update_args, StoreDictKeyPair
from common import constants as c
from aicrowd.aicrowd_utils import is_on_aicrowd_server
import models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


def main(args):
    on_aicrowd_server = is_on_aicrowd_server()

    if on_aicrowd_server:
        from aicrowd import aicrowd_helpers
        aicrowd_helpers.execution_start()
        aicrowd_helpers.register_progress(0.)

        # turn off any logging
        args.use_wandb = False
        args.all_iter = args.max_iter + 1

    model_cl = getattr(models, args.alg)
    model = model_cl(args)
    if args.ckpt_load:
        model.load_checkpoint(args.ckpt_load, load_iternum=args.ckpt_load_iternum, load_optim=args.ckpt_load_optim)

    if not args.test:
        model.train()
    else:
        model.test()

    if args.aicrowd_challenge:
        from aicrowd import utils_pytorch as pyu, aicrowd_helpers
        # Export the representation extractor
        path_to_saved = pyu.export_model(pyu.RepresentationExtractor(model.model.encoder, 'mean'),
                                         input_shape=(1, model.num_channels, model.image_size, model.image_size))
        logging.info('A copy of the model saved in {}'.format(path_to_saved))

        if on_aicrowd_server:
            aicrowd_helpers.register_progress(1.0)
            aicrowd_helpers.submit()
        else:
            # The local_evaluation is implemented by aicrowd in the global namespace, so importing it suffices.
            from aicrowd import local_evaluation


def get_args(sys_args):
    parser = argparse.ArgumentParser(description='disentanglement-pytorch')

    # NeurIPS2019 AICrowd Challenge
    parser.add_argument('--aicrowd_challenge', default=False, type=str2bool, help='Run is an AICrowd submission')
    parser.add_argument('--evaluate_metric', default=None, type=str, choices=c.EVALUATION_METRICS, nargs='+',
                        help='Metric to evaluate the model during training')

    # name
    parser.add_argument('--alg', type=str, help='the disentanglement algorithm', choices=c.ALGS)
    parser.add_argument('--vae_loss', help='type of VAE loss', default=c.VAE_LOSS[0], choices=c.VAE_LOSS)
    parser.add_argument('--vae_type', help='type of VAE', nargs='*', default=c.VAE_TYPE[0], choices=c.VAE_TYPE)
    parser.add_argument('--name', default='unknown_experiment', type=str, help='name of the experiment')

    # Neural architectures
    parser.add_argument('--encoder', type=str, nargs='+', required=True, choices=c.ENCODERS,
                        help='name of the encoder network')
    parser.add_argument('--decoder', type=str, nargs='+', required=True, choices=c.DECODERS,
                        help='name of the decoder network')
    parser.add_argument('--label_tiler', type=str, nargs='*', choices=c.TILERS,
                        help='the tile network used to convert one hot labels to 2D channels')
    parser.add_argument('--discriminator', type=str, nargs='*', choices=c.DISCRIMINATORS,
                        help='the discriminator network')

    # Test or train
    parser.add_argument('--test', default=False, type=str2bool, help='to test')

    # training hyper-params
    parser.add_argument('--max_iter', default=3e7, type=float, help='maximum training iteration')
    parser.add_argument('--max_epoch', default=3e7, type=float, help='maximum training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_disc_layers', default=5, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_disc_layers', default=1000, type=int, help='size of fc layers in discriminators')

    # latent encoding
    parser.add_argument('--z_dim', default=16, type=int, help='size of the encoded z space')
    parser.add_argument('--include_labels', default=None, type=str, nargs='*',
                        help='Labels (indices or names) to include in latent encoding.')
    parser.add_argument('--l_dim', default=0, type=str, help='size of the encoded w space (for each label)')

    # optimizer
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 parameter of the Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 parameter of the Adam optimizer')
    parser.add_argument('--lr_G', default=1e-4, type=float, help='learning rate of the main autoencoder')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of all the discriminators')

    # Neural architectures hyper-parameters
    parser.add_argument('--num_layer_disc', default=6, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_layer_disc', default=1000, type=int, help='size of fc layers in discriminators')

    # Loss weights and parameters [Common]
    parser.add_argument('--w_recon', default=1.0, type=float, help='reconstruction loss weight')
    parser.add_argument('--w_kld', default=1.0, type=float, help='main KLD loss weight (e.g. in BetaVAE)')

    # Loss weights and parameters for [CapacityVAE]
    parser.add_argument('--max_c', default=25.0, type=float, help='maximum value of control parameter in CapacityVAE')
    parser.add_argument('--iterations_c', default=100000, type=int, help='how many iterations to reach max_c')

    # Loss weights and parameters for [FactorVAE]
    parser.add_argument('--w_tc_empirical', default=1.0, type=float,
                        help='total correlation loss weight (e.g. in FactorVAE)')

    # Loss weights and parameters for [BetaTCVAE]
    parser.add_argument('--w_tc_analytical', default=1.0, type=float,
                        help='total correlation loss weight (e.g. in BetaTCVAE)')

    # Loss weights and parameters for [InfoVAE]
    parser.add_argument('--w_infovae', default=1.0, type=float,
                        help='mmd loss weight (e.g. in InfoVAE)')

    # Loss weights and parameters for [DIPVAE]
    parser.add_argument('--w_dipvae', default=1.0, type=float,
                        help='covariance regularizer loss weight (e.g. in DIPVAE I and II)')

    # Loss weights and parameters for [IFCVAE]
    parser.add_argument('--w_le', default=1.0, type=float, help='label encoding loss weight (e.g. in IFCVAE)')
    parser.add_argument('--w_aux', default=1.0, type=float, help='auxiliary discriminator loss weight (e.g. in IFCVAE)')

    # Hyperparameters for [DIP-VAE]
    parser.add_argument('--lambda_d_factor', default=10.0, type=float,
                        help='Hyperparameter for diagonal values of covariance matrix')
    parser.add_argument('--lambda_od', default=1.0, type=float,
                        help='Hyperparameter for off diagonal values of covariance matrix.')
    parser.add_argument('--dip_type', default='i', type=str, choices=['i', 'ii'],
                        help='Type of DIP-VAE.')

    # Dataset
    parser.add_argument('--dset_dir', default=os.getenv('DISENTANGLEMENT_LIB_DATA', './data'),
                        type=str, help='main dataset directory')
    parser.add_argument('--dset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='width and height of image')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers for the data loader')

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
    parser.add_argument('--ckpt_load_optim', default=True, type=str2bool, help='load the optimizer state')

    # Iterations [default for all is equal to 1 epoch]
    parser.add_argument('--ckpt_save_iter', default=None, type=int, help='iters to save checkpoint '
                                                                         '[default: 1 epoch]')
    parser.add_argument('--evaluate_iter', default=None, type=int, help='iters to evaluate the disentanglement '
                                                                        '[default: 1 epoch]')
    parser.add_argument('--float_iter', default=None, type=int, help='iters to aggregate float logs '
                                                                     '[default: 1 epoch]')
    parser.add_argument('--recon_iter', default=None, type=int, help='iters to reconstruct input image '
                                                                     '[default: 1 epoch]')
    parser.add_argument('--traverse_iter', default=None, type=int, help='iters to traverse and visualize latent spaces '
                                                                        '[default: 1 epoch]')
    parser.add_argument('--print_iter', default=None, type=int, help='iters to print float values '
                                                                     '[default: 1 epoch]')
    parser.add_argument('--all_iter', default=None, type=int, help='use same iteration for all '
                                                                   '[default: 1 epoch]')

    # Learning rate scheduler
    parser.add_argument('--lr_scheduler', default=None, type=str, choices=c.LR_SCHEDULERS,
                        help='Type of learning rate scheduler [default: no scheduler]')
    parser.add_argument("--lr_scheduler_args", dest='lr_scheduler_args', action=StoreDictKeyPair, nargs="+",
                        metavar="KEY=VAL", help="Arguments of the for the lr_scheduler. See PyTorch docs.")

    args = parser.parse_args(sys_args)

    assert args.image_size == 64, 'for now, models are hard coded to support only image size of 64x64'

    args.num_labels = 0
    if args.include_labels is not None:
        args.num_labels = len(args.include_labels)

    # makedirs
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)
    assert os.path.exists(args.dset_dir), 'Main dataset directory does not exist at {}'.format(args.dset_dir)

    # test
    args = update_args(args) if args.test else args

    # todo: check wandb import and turn it off it fails

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
