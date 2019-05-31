"""utils.py"""

import os
import logging
import argparse
import subprocess
import scipy.linalg as linalg
import numpy as np

import torch.nn.functional as F
import random


class DataGatherer(object):
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg: [] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


def str2list(v, type):
    if v is None:
        return None
    return [type(item) for item in v.split(',')]


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay ' + str(delay) + ' -loop 0 ' + image_str + ' ' + output_gif
    subprocess.call(str1, shell=True)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : mu of dist 1
    -- mu2   : mu of dist 2
    -- sigma1: The covariance matrix of dist 1
    -- sigma2: The covariance matrix of dist 2
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class LinearScheduler:
    def __init__(self, start_value, target_value, total_iters):
        self.start_value = start_value
        self.target_value = target_value
        self.total_iters = total_iters

        self.inc_per_iter = (target_value - start_value) / total_iters

    def get(self, iter):
        return self.start_value + iter * self.inc_per_iter


def dataset_samples(dataset, device):
    sample_idx = {}
    dset_name = dataset.name
    if dset_name.lower() == 'dsprites':
        fixed_idx = [87040, 332800, 578560]  # square ellipse heart
        sample_idx = {'fixed_square': fixed_idx[0],
                      'fixed_ellipse': fixed_idx[1],
                      'fixed_heart': fixed_idx[2]
                      }

    elif dset_name.lower() == 'celeba':
        fixed_idx = [11281, 114307, 10535, 59434]
        sample_idx = {'fixed_1': fixed_idx[0],
                      'fixed_2': fixed_idx[1],
                      'fixed_3': fixed_idx[2],
                      'fixed_4': fixed_idx[3],
                      }
    else:
        for i in range(3):
            randidx = random.randint(0, dataset.__len__())
            sample_idx.update({'rand' + str(i): randidx})

    # add a random sample to all
    randidx = random.randint(0, dataset.__len__())
    sample_idx.update({'random': randidx})

    images = {}
    labels = {}
    for key, idx in sample_idx.items():
        sample = dataset.__getitem__(idx)

        img = sample[0]
        images[key] = img.to(device).unsqueeze(0)

        if dataset.labels is not None:
            label = sample[2]
            labels[key] = label.to(device).unsqueeze(0)

    return images, labels


class StyleFormatter(logging.Formatter):
    CSI = "\x1B["
    YELLOW = '33;40m'
    RED = '31;40m'

    high_style = '{}{}(%(levelname)s)[%(asctime)s %(filename)s:%(lineno)d]  %(message)s{}0m'.format(CSI, RED, CSI)
    medium_style = '{}{}(%(levelname)s)[%(asctime)s %(filename)s:%(lineno)d]  %(message)s{}0m'.format(CSI, YELLOW, CSI)
    low_style = '(%(levelname)s)[%(asctime)s %(filename)s:%(lineno)d]  %(message)s'

    def __init__(self, fmt=None, datefmt='%b-%d %H:%M', style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        if record.levelno <= logging.INFO:
            self._style = logging.PercentStyle(StyleFormatter.low_style)
        elif record.levelno <= logging.WARNING:
            self._style = logging.PercentStyle(StyleFormatter.medium_style)
        else:
            self._style = logging.PercentStyle(StyleFormatter.high_style)

        return logging.Formatter.format(self, record)


if __name__ == '__main__':
    a = np.random.rand(10) / 10
    b = np.random.rand(10) / 10
    astd = np.eye(10) * 10
    bstd = np.eye(10) * 10
    print(frechet_distance(a, astd, b, bstd))
