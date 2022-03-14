import logging
import argparse
import subprocess
from sklearn.linear_model import LinearRegression, MultiTaskLasso, HuberRegressor
import scipy.linalg as linalg
import numpy as np
import random
from importlib import reload
import os

import torch.nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from common import constants as c


def cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


def str2bool(v):
    """
    Thank to stackoverflow user: Maxim
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    :param v: A command line argument with values [yes, true, t, y, 1, True, no, false, f, n, 0, False]
    :return: Boolean version of the command line argument
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def grid2gif(image_str, output_gif, delay=100):
    """
    Makes an animated GIF from input images.
    Thanks to the stackoverflow user: pkjain (https://stackoverflow.com/users/5241303/pkjain)

    :param image_str: The wildcard path to all images whose names are numerically sorted
    :param output_gif: Path to the generated GIF
    :param delay: Delay to introduce in-between showing consecutive images of a GIF
    """

    str1 = 'convert -delay {} -loop 0 {} {}'.format(delay, image_str, output_gif)
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
    def __init__(self, start_value, target_value=None, epochs=None):
        self.start_value = start_value
        self.target_value = target_value
        assert start_value != target_value, 'start_value and target_value should be different'
        self.mode = min if target_value > start_value else max
        self.per_step = (target_value - start_value) / epochs

    def step(self, step_num):
        return self.mode(self.start_value + step_num * self.per_step, self.target_value)


def get_data_for_visualization(dataset, device):
    def random_idx():
        return random.randint(0, dataset.__len__() - 1)

    sample_idx = {}
    dset_name = dataset.name
    if dset_name == c.DATASETS[1]:  # dsprites_full
        fixed_idx = [87040, 332800, 578560]  # square ellipse heart
        sample_idx = {'{}_{}'.format(c.FIXED, c.SQUARE): fixed_idx[0],
                      '{}_{}'.format(c.FIXED, c.ELLIPSE): fixed_idx[1],
                      '{}_{}'.format(c.FIXED, c.HEART): fixed_idx[2]
                      }

    elif dset_name == c.DATASETS[0]:  # celebA
        fixed_idx = [11281, 114307, 10535, 59434]
        sample_idx = {}
        for i in range(len(fixed_idx)):
            sample_idx.update({c.FIXED + '_' + str(i): fixed_idx[i]})

    else:
        for i in range(3):
            sample_idx.update({'rand' + str(i): random_idx()})

    # add a random sample to all
    sample_idx.update({c.RANDOM: random_idx()})

    images = {}
    labels = {}
    for key, idx in sample_idx.items():
        try:
            data = dataset.__getitem__(idx)
        except IndexError:
            data = dataset.__getitem__(random_idx())

        images[key] = data[0].to(device).unsqueeze(0)

        labels[key] = None
        if dataset.has_labels():
            labels[key] = data[1].to(device, dtype=torch.long).unsqueeze(0)

    return images, labels


def prepare_data_for_visualization(data):
    sample_images, sample_labels = data
    sample_images_dict = {}
    sample_labels_dict = {}
    for i, img in enumerate(sample_images):
        sample_images_dict.update({str(i): img})
        if sample_labels is not None:
            sample_labels_dict.update({str(i): sample_labels[i].to(dtype=torch.long).unsqueeze(0)})
        else:
            sample_labels_dict.update({str(i): None})

    return sample_images_dict, sample_labels_dict



class StyleFormatter(logging.Formatter):
    CSI = "\x1B["
    YELLOW = '33;40m'
    RED = '31;40m'

    # Add %(asctime)s after [ to include the time-date of the log
    high_style = '{}{}(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s{}0m'.format(CSI, RED, CSI)
    medium_style = '{}{}(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s{}0m'.format(CSI, YELLOW, CSI)
    low_style = '(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s'

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


def _init_layer(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
    if isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight.data)


def init_layers(modules):
    for block in modules:
        from collections.abc import Iterable
        if isinstance(modules[block], Iterable):
            for m in modules[block]:
                _init_layer(m)
        else:
            _init_layer(modules[block])


@static_var("Ys", dict())
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N, NumLabels].
      num_classes: list of int for number of classes for each label, or single label (int)

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    if isinstance(num_classes, int):
        num_classes = [num_classes]

    one_hots = []
    for i in range(len(num_classes)):
        num_class = num_classes[i]

        if num_class not in one_hot_embedding.Ys:
            one_hot_embedding.Ys[num_class] = cuda(torch.eye(num_class))

        y = one_hot_embedding.Ys[num_class]
        one_hots.append(y[labels[:, i]])

    return torch.cat(one_hots, dim=1)


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                my_dict[k] = float(v)
            except ValueError:
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def get_scheduler(value, scheduler_type, scheduler_args):
    if scheduler_type is None:
        return
    if isinstance(value, torch.optim.Optimizer):
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
        scheduler = scheduler_class(value, **scheduler_args)
    else:
        if scheduler_type == 'LinearScheduler':
            scheduler = LinearScheduler(value, **scheduler_args)
        else:
            raise NotImplementedError

    return scheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def is_time_for(iteration, milestone):
    return iteration % milestone == milestone - 1


def setup_logging(verbose):
    # verbosity
    reload(logging)  # to turn off any changes to logging done by other imported libraries
    h = logging.StreamHandler()
    h.setFormatter(StyleFormatter())
    h.setLevel(0)
    logging.root.addHandler(h)
    logging.root.setLevel(verbose)


def initialize_seeds(seed):
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def set_environment_variables(dset_dir, dset_name):
    """
    If the argument dset_dir is set, overwrite DISENTANGLEMENT_LIB_DATA.
    else if only $DATASETS is set, use the same for $DISENTANGLEMENT_LIB_DATA
    else if only $DISENTANGLEMENT_LIB_DATA is set, use the same for $DATASETS
    else print warning that the environment variables are not set or inconsistent.

    If the argument dset_name is set, overwrite $DATASET_NAME.
    else if only $DATASET_NAME is set, use the same for $AICROWD_DATASET_NAME
    else if only $AICROWD_DATASET_NAME is set, use the same for $DATASET_NAME
    else print warning that the environment variables are not set or inconsistent.

    :param dset_dir: directory where all the datasets are saved
    :param dset_name: name of the dataset to be loaded by the dataloader
    """
    if dset_dir:
        os.environ['DISENTANGLEMENT_LIB_DATA'] = dset_dir
    if not os.environ.get('DISENTANGLEMENT_LIB_DATA'):
        logging.warning(f"Environment variables are not correctly set:\n"
                        f"$DISENTANGLEMENT_LIB_DATA={os.environ.get('DISENTANGLEMENT_LIB_DATA')}\n")

    if dset_name:
        os.environ['DATASET_NAME'] = dset_name
    if os.environ.get('DATASET_NAME') and not os.environ.get('AICROWD_DATASET_NAME'):
        os.environ['AICROWD_DATASET_NAME'] = os.getenv('DATASET_NAME')
    elif os.environ.get('AICROWD_DATASET_NAME') and not os.environ.get('DATASET_NAME'):
        os.environ['DATASET_NAME'] = os.getenv('AICROWD_DATASET_NAME')
    elif os.environ.get('AICROWD_DATASET_NAME') != os.environ.get('DATASET_NAME'):
        logging.warning(f"Environment variables are not correctly set:\n"
                        f"$AICROWD_DATASET_NAME={os.environ.get('AICROWD_DATASET_NAME')}\n"
                        f"$DATASET_NAME={os.environ.get('DATASET_NAME')}")

    logging.info(f"$AICROWD_DATASET_NAME={os.environ.get('AICROWD_DATASET_NAME')}")
    logging.info(f"$DATASET_NAME={os.environ.get('DATASET_NAME')}")
    logging.info(f"$DISENTANGLEMENT_LIB_DATA={os.environ.get('DISENTANGLEMENT_LIB_DATA')}")


def make_dirs(args):
    # makedirs
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)


## ADD F1_SCORE

class F1_Loss(torch.nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, ):
#        assert y_pred.ndim == 2
 #       assert y_true.ndim == 1
        y_true = F.one_hot(y_true.to(dtype = torch.long), 2).to(torch.float32)
        #y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

## ADD ACCURACY

class Accuracy_Loss(torch.nn.Module):
    '''Calculate Accuracy score. Can work with gpu tensors

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, ):
        #print(y_pred.size(1))
        #print(y_true.size())
        assert y_pred.size(1) == 2
        #assert y_true.size(1) == 1

        y_true = F.one_hot(y_true.to(dtype = torch.long), 2).to(torch.float32)
        #y_pred = F.softmax(y_pred, dim=1)

#        ymask = (y_pred[:,0] > 0.5)
 #       y_pred[ymask][:,0], y_pred[ymask][:,1] = 1, 0
  #      y_pred[~ymask][:,0], y_pred[~ymask][:,1] = 0,1

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        acc = (tp + tn)/(tp + tn + fp + fn)

#        acc = acc.clamp(min=self.epsilon, max=1 - self.epsilon)

        return acc.mean()


def Interpretability(z, g, rel_factors=10 ** 4):
    '''
    Compute the interpretability score of the latent factors given the generative ones.
    It can be done on a restricted number of entries.
    '''

    D = len(z[0])
    K = len(g[0])

    model = HuberRegressor()
    coeff = np.zeros(shape=(D, K))
    total_acc = 0
    for i in range(K):
        model.fit(z, g[:, i])
        coeff[:, i] = model.coef_
        total_acc += model.score(z, g[:, i])

    print('# Interpret score # Total accuracy of Regressor model:', total_acc / K)
    # print('Regression score:',model.score(z,g)*100,'%')

    R = np.abs(coeff)

    R_g = np.sum(R, axis=0)

    P = np.zeros(np.shape(R))
    for i in range(D):
        for j in range(K):
            P[i, j] = R[i, j] / (R_g[j] + 10 ** -6)

    H_D = np.zeros(K)
    for j in range(K):
        P[:, j] += 10 ** -6
        for i in range(D):
            if P[i, j] > 1: P[i, j] = 1
        H_D[j] = - np.sum(P[:, j] * np.log(P[:, j]))

    I = 1 - H_D / np.log(D)

    rho = np.sum(R, axis=0) / np.sum(R + 10 ** -6)

    I_tot = np.sum(rho * I)

    return I, I_tot



def old_Interpret(z,g, rel_factors=1000):
    D = len(z[0])
    K = len(g[0])

    z = z[:rel_factors]
    g = g[:rel_factors]

    # lasso variant
    model = MultiTaskLasso()
    model.fit(z, g)
#    print('Model weights', model.coef_)

    print('Fitting the generative part with:', model.score(z, g,)*100,'% accuracy')

    R = np.abs(model.coef_) # R \in R^K*D
    R_g = np.sum(R, axis=0) # R_g \in R^D

    P = np.zeros(np.shape(R))
    for i in range(K):
        for j in range(D):
            P[i, j] = R[i, j] / (R_g[j] + 10 ** -6)

    H_D = np.zeros(K)
    for j in range(K):
        H_D[j] = - np.sum(P[:, j] * np.log(P[:, j] + 10 ** -6))

    I = 1 - H_D / np.log(D)

    rho = np.sum(R, axis=1) / np.sum(R)

    I_tot = np.sum(rho * I)

    return I, I_tot