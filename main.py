import sys

import json
import pandas as pd
import torch
import logging
import os
import datetime
from common.utils import setup_logging, initialize_seeds, set_environment_variables
from aicrowd.aicrowd_utils import is_on_aicrowd_server
from common.arguments import get_args
import models

torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(_args):
    #
    # REMOVED AIRCROWD CHALLENGE HERE
    #

    #include path for saving model performances

    dset_name = _args.dset_name
    nowstr = datetime.datetime.now().strftime("%Y_%m_%d-%H")

#    data_args = pd.DataFrame(vars(_args) )#list(_args.values()), index=_args.keys())
 #   data_args.to_csv(out_path, index=False)

    # load the model associated with args.alg
    print("_Args.alg", _args.alg)

    model_cl = getattr(models, _args.alg)
    model = model_cl(_args) #return models.alg

    # load checkpoint
    if _args.ckpt_load:
        model.load_checkpoint(_args.ckpt_load, load_iternum=_args.ckpt_load_iternum, load_optim=_args.ckpt_load_optim)

    # run test or train



    if not _args.test:
        if _args.out_path is not None:
            out_path = os.path.join("logs/", f"{dset_name}__{_args.alg}__{nowstr}")
            if not os.path.exists(os.path.join(out_path, 'train_runs')):
                os.makedirs(os.path.join(out_path, 'train_runs'))

            # SAVE _args
            print(type(_args))
            with open(os.path.join(out_path, 'commandline_args.txt'), 'w') as f:
                json.dump(_args.__dict__, f, indent=2)

            model.train(output=os.path.join(out_path))
        else:
            model.train()
    else:
        model.test()



    ### REMOVED AIRCROWD ###

if __name__ == "__main__":
    _args = get_args(sys.argv[1:])
    setup_logging(_args.verbose)
    initialize_seeds(_args.seed)

    # set the environment variables for dataset directory and name, and check if the root dataset directory exists.
    set_environment_variables(_args.dset_dir, _args.dset_name)
    assert os.path.exists(os.environ.get('DISENTANGLEMENT_LIB_DATA', '')), \
        'Root dataset directory does not exist at: \"{}\"'.format(_args.dset_dir)

    main(_args)
