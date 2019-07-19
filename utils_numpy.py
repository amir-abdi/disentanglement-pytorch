import os
import dill

import numpy as np


def get_function_path(base_path=None, experiment_name=None, make=True):
    """
    This function gets the path to where the function is expected to be stored.

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
    base_path = os.getenv("AICROWD_OUTPUT_PATH","../scratch/shared") \
        if base_path is None else base_path
    experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name") \
        if experiment_name is None else experiment_name
    model_path = os.path.join(base_path, experiment_name, 'representation', 'python_model.dill')
    if make:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(model_path), 'results'), exist_ok=True)
    return model_path


def export_function(fn, path=None):
    """
    Exports a function. This tries to serialize the argument `fn`, which must be callable
    and expect as input a numpy tensor of shape NCHW, where N (batch-size) can be arbitrary,
    C (channel) is the number of input channels, and (H, W) are the dimensions of the image.

    There are no guarantees that the serialization works as expected - you should double
    check that this is indeed the case by importing the function.

    Parameters
    ----------
    fn : callable
        Function to be serialized.
    path : str
        Path to the file where the function is saved. Defaults to the value set by the
        `get_model_path` function above.

    Returns
    -------
    str
        Path to where the function is saved.
    """
    assert callable(fn), "Provided function should at least be callable..."
    path = get_function_path() if path is None else path
    with open(path, 'wb') as f:
        dill.dump(fn, f, protocol=dill.HIGHEST_PROTOCOL)
    return path


def import_function(path=None):
    """
    Imports a function from file.

    Parameters
    ----------
    path : str
        Path to where the function is saved. Defaults to the return value of `get_function_path`
        function defined above.

    Returns
    -------
    callable
    """
    path = get_function_path() if path is None else path
    with open(path, 'rb') as f:
        # Here goes nothing...
        fn = dill.load(f)
    return fn


def make_representor(fn, format='NCHW'):
    """
    Wraps a function in another callable that can be used by `disentanglement_lib`.

    Parameters
    ----------
    fn : callable
        Function to be wrapped.
    format : str
        Input format expected by `fn`. Can be NCHW or NHWC, where
            N: batch
            C: channels
            H: height
            W: width

    Returns
    -------
    callable
    """
    assert format in ['NCHW', 'NHWC'], "format must either be NCHW or NHWC; got {format}."

    def _represent(x):
        assert isinstance(x, np.ndarray), \
            "Input to the representation function must be a ndarray, got {type(x)} instead."
        assert x.ndim == 4, \
            "Input to the representation function must be a four dimensional NHWC array, " \
            "got a {x.ndim}-dimensional array of shape {x.shape} instead."
        # Convert from NHWC to NCHW
        if format == 'NCHW':
            x = np.moveaxis(x, 3, 1)
            N, C, H, W = x.shape
        else:
            N, H, W, C = x.shape
        # Call the function on the array and validate its shape
        y = fn(x)
        assert isinstance(y, np.ndarray), "Output from the representation function " \
            "should be a numpy array, got {type(y)} instead."
        assert y.ndim == 2, "Output from the representation function should be two dimensional."
        return y

    return _represent


