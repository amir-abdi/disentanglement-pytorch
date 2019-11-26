import sys
import torch
import logging
import os

from common.utils import setup_logging, initialize_seeds, set_environment_variables
from aicrowd.aicrowd_utils import is_on_aicrowd_server
from common.arguments import get_args
import models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(_args):
    on_aicrowd_server = is_on_aicrowd_server()

    if on_aicrowd_server:
        from aicrowd import aicrowd_helpers
        aicrowd_helpers.execution_start()
        aicrowd_helpers.register_progress(0.)

        # turn off all logging and visualization
        _args.use_wandb = False
        _args.all_iter = _args.max_iter + 1

    # load the model associated with args.alg
    model_cl = getattr(models, _args.alg)
    model = model_cl(_args)

    # load checkpoint
    if _args.ckpt_load:
        model.load_checkpoint(_args.ckpt_load, load_iternum=_args.ckpt_load_iternum, load_optim=_args.ckpt_load_optim)

    # run test or train
    if not _args.test:
        model.train()
    else:
        model.test()

    # if this is part of the aicrowd_challenge, export and inference model and run disentanglement evaluation
    if _args.aicrowd_challenge:
        from aicrowd import utils_pytorch as pyu, aicrowd_helpers
        # Export the representation extractor
        path_to_saved = pyu.export_model(pyu.RepresentationExtractor(model.model.encoder, 'mean'),
                                         input_shape=(1, model.num_channels, model.image_size, model.image_size))
        logging.info(f'A copy of the model saved in {path_to_saved}')

        if on_aicrowd_server:
            # AICrowd will handle the evaluation
            aicrowd_helpers.register_progress(1.0)
            aicrowd_helpers.submit()
        else:
            # Run evaluation locally
            # The local_evaluation is implemented by aicrowd in the global namespace, so importing it suffices.
            #  todo: implement a modular version of local_evaluation
            # noinspection PyUnresolvedReferences
            from aicrowd import local_evaluation


if __name__ == "__main__":
    _args = get_args(sys.argv[1:])
    setup_logging(_args.verbose)
    initialize_seeds(_args.seed)

    # set the environment variables for dataset directory and name, and check if the root dataset directory exists.
    set_environment_variables(_args.dset_dir, _args.dset_name)
    assert os.path.exists(os.environ.get('DISENTANGLEMENT_LIB_DATA', '')), \
        'Root dataset directory does not exist at: \"{}\"'.format(_args.dset_dir)

    main(_args)
