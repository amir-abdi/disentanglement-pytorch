"""
Borrowed from https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/utils_pytorch.py
"""

from copy import deepcopy
from collections import namedtuple

import numpy as np
import torch
from torch.jit import trace

# ------ Data Loading ------
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import os
if 'DISENTANGLEMENT_LIB_DATA' not in os.environ:
    os.environ.update({'DISENTANGLEMENT_LIB_DATA': os.path.join(os.path.dirname(__file__),
                                                                'scratch',
                                                                'dataset')})
# noinspection PyUnresolvedReferences
from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

from common.data_loader import get_dataset_name


ExperimentConfig = namedtuple('ExperimentConfig',
                              ('base_path', 'experiment_name', 'dataset_name'))


def get_config():
    """
    This function reads the environment variables AICROWD_OUTPUT_PATH,
    AICROWD_EVALUATION_NAME and AICROWD_DATASET_NAME and returns a
    named tuple.
    """
    return ExperimentConfig(base_path=os.getenv("AICROWD_OUTPUT_PATH", "./scratch/shared"),
                            experiment_name=os.getenv("AICROWD_EVALUATION_NAME", "experiment_name"),
                            dataset_name=os.getenv("AICROWD_DATASET_NAME", "cars3d"))


def use_cuda():
    """
    Whether to use CUDA for evaluation. Returns True if CUDA is available and
    the environment variable AICROWD_CUDA is not set to False.
    """
    return torch.cuda.is_available() and os.getenv('AICROWD_CUDA', True)


def get_model_path(base_path=None, experiment_name=None, make=True):
    """
    This function gets the path to where the model is expected to be stored.

    Parameters
    ----------
    base_path : str
        Path to the directory where the experiments are to be stored.
        This defaults to AICROWD_OUTPUT_PATH (see `get_config` above) and which in turn
        defaults to './scratch/shared'.
    experiment_name : str
        Name of the experiment. This defaults to AICROWD_EVALUATION_NAME which in turn
        defaults to 'experiment_name'.
    make : Makes the directory where the returned path leads to (if it doesn't exist already)

    Returns
    -------
    str
        Path to where the model should be stored (to be found by the evaluation function later).
    """
    base_path = os.getenv("AICROWD_OUTPUT_PATH", "../scratch/shared") \
        if base_path is None else base_path
    experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name") \
        if experiment_name is None else experiment_name
    model_path = os.path.join(base_path, experiment_name, 'representation', 'pytorch_model.pt')
    if make:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(model_path), 'results'), exist_ok=True)
    return model_path


def export_model(model, path=None, input_shape=(1, 3, 64, 64)):
    """
    Exports the model. If the model is a `ScriptModule`, it is saved as is. If not,
    it is traced (with the given input_shape) and the resulting ScriptModule is saved
    (this requires the `input_shape`, which defaults to the competition default).

    Parameters
    ----------
    model : torch.nn.Module or torch.jit.ScriptModule
        Pytorch Module or a ScriptModule.
    path : str
        Path to the file where the model is saved. Defaults to the value set by the
        `get_model_path` function above.
    input_shape : tuple or list
        Shape of the input to trace the module with. This is only required if model is not a
        torch.jit.ScriptModule.

    Returns
    -------
    str
        Path to where the model is saved.
    """
    path = get_model_path() if path is None else path
    model = deepcopy(model).cpu().eval()
    if not isinstance(model, torch.jit.ScriptModule):
        assert input_shape is not None, "`input_shape` must be provided since model is not a " \
                                        "`ScriptModule`."
        traced_model = trace(model, torch.zeros(*input_shape))
    else:
        traced_model = model
    torch.jit.save(traced_model, path)
    return path


def import_model(path=None):
    """
    Imports a model (as ScriptModule) from file.

    Parameters
    ----------
    path : str
        Path to where the model is saved. Defaults to the return value of the `get_model_path`
        function above.

    Returns
    -------
    torch.jit.ScriptModule
        The model file.
    """
    path = get_model_path() if path is None else path
    return torch.jit.load(path)


def make_representor(model, cuda=None):
    """
    Encloses the pytorch ScriptModule in a callable that can be used by `disentanglement_lib`.

    Parameters
    ----------
    model : torch.nn.Module or torch.jit.ScriptModule
        The Pytorch model.
    cuda : bool
        Whether to use CUDA for inference. Defaults to the return value of the `use_cuda`
        function defined above.

    Returns
    -------
    callable
        A callable function (`representation_function` in dlib code)
    """
    # Deepcopy doesn't work on ScriptModule objects yet:
    # https://github.com/pytorch/pytorch/issues/18106
    # model = deepcopy(model)
    cuda = use_cuda() if cuda is None else cuda
    model = model.cuda() if cuda else model.cpu()

    # Define the representation function
    def _represent(x):
        assert isinstance(x, np.ndarray), \
            "Input to the representation function must be a ndarray."
        assert x.ndim == 4, \
            "Input to the representation function must be a four dimensional NHWC tensor."
        # Convert from NHWC to NCHW
        x = np.moveaxis(x, 3, 1)
        # Convert to torch tensor and evaluate
        x = torch.from_numpy(x).float().to('cuda' if cuda else 'cpu')
        with torch.no_grad():
            y = model(x)
        y = y.cpu().numpy()
        assert y.ndim == 2, \
            "The returned output from the representor must be two dimensional (NC)."
        return y

    return _represent


class RepresentationExtractor(torch.nn.Module):
    VALID_MODES = ['mean', 'sample']

    def __init__(self, encoder, mode='mean'):
        super(RepresentationExtractor, self).__init__()
        assert mode in self.VALID_MODES, '`mode` must be one of {self.VALID_MODES}'
        self.encoder = encoder
        self.mode = mode

    def forward(self, x):
        mu, logvar = self.encoder(x)
        if self.mode == 'mean':
            return mu
        elif self.mode == 'sample':
            return self.reparameterize(mu, logvar)
        else:
            raise NotImplementedError

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


if __name__ == '__main__':
    pass
