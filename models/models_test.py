import pytest
import os
import shutil
import itertools
import logging

import models
from main import get_args
from common import constants as c
from common.arguments import update_args

ALGS = c.ALGS
VAE_LOSSES = c.LOSS_TERMS
DATASETS = c.TEST_DATASETS  # 'celebA', 'dsprites_full'
BATCH_SIZE = 4
MAX_ITER = 6
CKPT_SAVE_ITER = 3
Z_DIM = 16
ALL_ITER = 3
NUM_SAMPLES = 12  # test datasets (dsprites and CelebA) each contain 12 samples


class TestModels(object):
    """
    Tests all the implemented algoritms defined in common.constants.ALGS
    with all the loss terms defined in common.constants.LOSS_TERMS.
    The mock data in the data/test_dsets folder is used for all tests.

    The tests are limited to end-to-end training of the models and logging/visualization plus
    saving and loading the models.
    The logic of VAE and their level of disentanglement are not tested.
    """
    @pytest.fixture(scope="module", params=itertools.product(DATASETS, ALGS))
    def args(self, request):
        dset_name = request.param[0]
        alg = request.param[1]
        sys_args = list(['--name=tmp_test',
                         '--alg={}'.format(alg),
                         '--dset_dir=./data/test_dsets',
                         '--dset_name={}'.format(dset_name),
                         '--z_dim={}'.format(Z_DIM),
                         '--batch_size={}'.format(BATCH_SIZE),
                         '--all_iter={}'.format(ALL_ITER),
                         '--evaluate_iter={}'.format(MAX_ITER*2),
                         '--ckpt_save_iter={}'.format(CKPT_SAVE_ITER),
                         '--max_iter={}'.format(MAX_ITER),
                         '--controlled_capacity_increase={}'.format('true'),
                         '--loss_terms'
                         ])
        sys_args.extend(VAE_LOSSES)

        encoder = (c.ENCODERS[1],)
        if alg == 'AE':
            encoder = (c.ENCODERS[0],)
        elif alg == 'IFCVAE':
            encoder = c.ENCODERS[1], c.ENCODERS[0]
        sys_args.append('--encoder')
        sys_args.extend(encoder)

        discriminator = (c.DISCRIMINATORS[0],)
        sys_args.append('--discriminator')
        sys_args.extend(discriminator)

        decoder = (c.DECODERS[0],)
        sys_args.append('--decoder')
        sys_args.extend(decoder)

        label_tiler = (c.TILERS[0],)
        sys_args.append('--label_tiler')
        sys_args.extend(label_tiler)

        if 'CVAE' in alg:
            if dset_name == c.DATASETS[1]:
                include_labels = '1', '2', '3'
            elif dset_name == 'celebA':
                include_labels = 'Wearing_Hat', 'Arched_Eyebrows'
            else:
                raise NotImplementedError
            sys_args.append('--include_labels')
            sys_args.extend(include_labels)

        args = get_args(sys_args)

        logging.info('sys_args', sys_args)
        logging.info('Testing {}:{}'.format(dset_name, alg))
        yield args

        # clean up: delete output and ckpt files
        train_dir = os.path.join(args.train_output_dir, args.name)
        test_dir = os.path.join(args.test_output_dir, args.name)
        ckpt_dir = os.path.join(args.ckpt_dir, args.name)

        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    def load_model(self, args):
        model_cl = getattr(models, args.alg)
        model = model_cl(args)
        return model

    @pytest.mark.dependency(name='test_train')
    def test_train(self, args):
        model = self.load_model(args)
        model.train()

        # test traverse and reconstructions
        train_dir = model.train_output_dir
        self.check_visualization_files_train(train_dir, args.dset_name)

        # test save checkpoint
        ckpt_dir = model.ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, 'last')
        assert os.path.exists(ckpt_path)

    @pytest.mark.dependency(depends=['test_train'])
    def test_load_ckpt(self, args):
        ckpt_path = os.path.join(args.ckpt_dir, args.name, 'last')
        args.ckpt_load = ckpt_path
        model = self.load_model(args)
        model.load_checkpoint(args.ckpt_load)

    @pytest.mark.dependency(depends=['test_train'])
    def test_test(self, args):
        args.test = True
        args = update_args(args)
        model = self.load_model(args)
        model.test()

        test_dir = model.test_output_dir
        self.check_visualization_files_test(test_dir)

    @staticmethod
    def check_visualization_files_train(output_dir, dset_name):
        assert os.path.exists(os.path.join(output_dir, '{}.{}'.format(c.RECON, c.JPG)))
        print('******', os.path.join(output_dir, '{}.{}'.format(c.RECON, c.JPG)))
        print('**********', os.path.join(output_dir, '{}_{}_0.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))

        if dset_name == c.DATASETS[0]:
            assert os.path.exists(os.path.join(output_dir, '{}_{}_0.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_1.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_2.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_3.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
        elif dset_name == c.DATASETS[1]:
            assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.SQUARE, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.ELLIPSE, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.HEART, c.JPG)))

        assert os.path.exists(os.path.join(output_dir, '{}_{}.{}'.format(c.TRAVERSE, c.RANDOM, c.JPG)))

    @staticmethod
    def check_visualization_files_test(output_dir):
        for b in range(NUM_SAMPLES // BATCH_SIZE):
            for s in range(BATCH_SIZE):
                assert os.path.exists(os.path.join(output_dir, '{}_{}.{}'.format(c.RECON, b, c.JPG)))
                assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, b, s, c.JPG)))
                assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, b, s, c.JPG)))
                assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, b, s, c.JPG)))
                assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, b, s, c.JPG)))
