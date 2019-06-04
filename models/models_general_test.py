import pytest
import os
import shutil
import itertools
import logging

import models
from main import get_args
from common import constants as c

DATASETS = ['celebA', 'dsprites']
ALGS = c.ALGS
VAE_LOSS = c.VAE_LOSS

class TestAE(object):
    @pytest.fixture(scope="module", params=itertools.product(DATASETS, ALGS, VAE_LOSS))
    def args(self, request):
        dset_name = request.param[0]
        alg = request.param[1]
        vae_loss = request.param[2]

        num_channels = 3
        if dset_name == 'dsprites':
            num_channels = 1
        encoder = 'SimpleGaussianEncoder64'
        if alg == 'AE':
            encoder='SimpleEncoder64'

        include_labels = ''
        if alg == 'CVAE':
            if dset_name == 'dsprites':
                include_labels = '1,2,3'
            elif dset_name == 'celebA':
                include_labels = 'Wearing_Hat,Arched_Eyebrows'

        args = get_args(['--name=tmp_test',
                         '--alg={}'.format(alg),
                         '--dset_dir=./data/test_dsets',
                         '--dset_name={}'.format(dset_name),
                         '--decoder=SimpleDecoder64',
                         '--encoder={}'.format(encoder),
                         '--num_channels={}'.format(num_channels),
                         '--z_dim=16',
                         '--batch_size=4',
                         '--all_iter=3',
                         '--ckpt_save_iter=3',
                         '--max_iter=4',
                         '--vae_loss={}'.format(vae_loss),
                         '--include_labels={}'.format(include_labels),
                         ])

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
    def test_train_color(self, args):
        model = self.load_model(args)
        model.train()

        # test traverse and reconstructions
        train_dir = model.train_output_dir
        assert os.path.exists(os.path.join(train_dir, '{}.{}'.format(c.RECONSTRUCTION, c.JPG)))
        if args.dset_name != 'dsprites':
            assert os.path.exists(os.path.join(train_dir, '{}_{}_0.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(train_dir, '{}_{}_1.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(train_dir, '{}_{}_2.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
            assert os.path.exists(os.path.join(train_dir, '{}_{}_3.{}'.format(c.TRAVERSE, c.FIXED, c.JPG)))
        else:
            assert os.path.exists(os.path.join(train_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.SQUARE, c.JPG)))
            assert os.path.exists(os.path.join(train_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.ELLIPSE, c.JPG)))
            assert os.path.exists(os.path.join(train_dir, '{}_{}_{}.{}'.format(c.TRAVERSE, c.FIXED, c.HEART, c.JPG)))

        assert os.path.exists(os.path.join(train_dir, '{}_{}.{}'.format(c.TRAVERSE, c.RANDOM, c.JPG)))

        # test save checkpoint
        ckpt_dir = model.ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, 'last')
        assert os.path.exists(ckpt_path)

    @pytest.mark.dependency(depends=['test_train_color'])
    def test_load_ckpt(self, args):
        ckpt_path = os.path.join(args.ckpt_dir, args.name, 'last')
        args.ckpt_load = ckpt_path
        model = self.load_model(args)
        model.load_checkpoint(args.ckpt_load)
