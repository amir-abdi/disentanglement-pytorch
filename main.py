import sys
import torch
import logging

from common.utils import setup_logging, initialize_seeds
from aicrowd.aicrowd_utils import is_on_aicrowd_server
from common.arguments import get_args
import models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


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
            # they will handle the evaluation
            aicrowd_helpers.register_progress(1.0)
            aicrowd_helpers.submit()
        else:
            # todo: implement a modular version of local_evaluation
            # The local_evaluation is implemented by aicrowd in the global namespace, so importing it suffices.
            # noinspection PyUnresolvedReferences
            from aicrowd import local_evaluation


if __name__ == "__main__":
    args_ = get_args(sys.argv[1:])
    setup_logging(args_.verbose)
    initialize_seeds(args_.seed)
    main(args_)
