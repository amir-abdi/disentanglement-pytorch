import pytest
import os
import shutil
import itertools
import logging

import models
from main import get_args
from common import constants as c
from common.utils import update_args

DATASETS = ['celebA', 'dsprites']
ALGS = c.ALGS
VAE_LOSS = c.VAE_LOSS


class TestModels(object):
    @pytest.fixture(scope="module", params=itertools.product(DATASETS, ALGS))
    def args(self, request):
        dset_name = request.param[0]
        alg = request.param[1]
        sys_args = list(['--name=tmp_test',
                         '--alg={}'.format(alg),
                         '--dset_dir=./data/test_dsets',
                         '--dset_name={}'.format(dset_name),
                         '--z_dim=16',
                         '--batch_size=4',
                         '--all_iter=3',
                         '--ckpt_save_iter=3',
                         '--max_iter=4',
                         '--vae_loss={}'.format('AnnealedCapacity'),
                         ])

        encoder = (c.ENCODERS[1],)
        if alg == 'AE':
            encoder = ('SimpleEncoder64',)
        elif alg == 'IFCVAE':
            encoder = 'SimpleGaussianEncoder64', 'SimpleEncoder64'
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
            if dset_name == 'dsprites':
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
        ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    def load_model(self, args):
        model_cl = getattr(models, args.alg)
        model = model_cl(args)
        return model

    @pytest.mark.dependency()
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
        assert os.path.exists(os.path.join(output_dir, '{}.{}'.format(c.RECONSTRUCTION, c.JPG)))
        if dset_name != 'dsprites':
            assert os.path.exists(os.path.join(output_dir, '{}_{}_0.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_1.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_2.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_3.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
        else:
            assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.SQUARE, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.ELLIPSE, c.JPG)))
            assert os.path.exists(os.path.join(output_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.HEART, c.JPG)))

        assert os.path.exists(os.path.join(output_dir, '{}_{}.{}'.format(c.TRAVERSE, c.RANDOM, c.JPG)))

    @staticmethod
    def check_visualization_files_test(output_dir):
        print(os.path.join(output_dir, '{}_0.{}'.format(c.RECONSTRUCTION, c.JPG)))
        assert os.path.exists(os.path.join(output_dir, '{}_0.{}'.format(c.RECONSTRUCTION, c.JPG)))
        assert os.path.exists(os.path.join(output_dir, '{}_0_0.{}'.format(c.TRAVERSE, c.JPG)))
        assert os.path.exists(os.path.join(output_dir, '{}_0_1.{}'.format(c.TRAVERSE, c.JPG)))
        assert os.path.exists(os.path.join(output_dir, '{}_0_2.{}'.format(c.TRAVERSE, c.JPG)))
        assert os.path.exists(os.path.join(output_dir, '{}_0_3.{}'.format(c.TRAVERSE, c.JPG)))
