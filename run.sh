#!/bin/bash

# Root is where this file is.
export NDC_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"

# Source the training environment (see the env variables defined therein) if we are not evaluating
if [ ! -n "${AICROWD_IS_GRADING+set}" ]; then
  # AICROWD_IS_GRADING is not set, so we're not running on the evaluator and it's safe to
  # source the train_environ.sh
  source ${NDC_ROOT}/setup_env.sh
else
  # We're on the evaluator.
  # Add root to python path, since this would usually be done in train_environ.sh
  export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}
fi

# If you have other dependencies, this would be a nice place to
# add them to your PYTHONPATH:
#export PYTHONPATH=${PYTHONPATH}:path/to/your/dependency

# Pytorch:
# 
# Note: In case of Pytorch, you will have to export your software runtime via 
#       Anaconda (After installing pytorch), as shown here : 
#		https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit#how-do-i-specify-my-software-runtime-
# 	as pytorch cannot be installed with just `pip`
#
export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}
python main.py \
--aicrowd_challenge=true \
--name=$NAME \
--alg=VAE \
--vae_loss=Basic \
--vae_type=FactorVAE \
--dset_dir=/home/amirabdi/data/Datasets/ \
--dset_name=dsprites \
--traverse_z=true \
--encoder=SimpleGaussianEncoder64 \
--decoder=SimpleDecoder64 \
--discriminator=SimpleDiscriminator \
--z_dim=10 \
--use_wandb=false \
--w_kld=1 \
--lr_G=0.0002 \
--max_iter=10000 \

#--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/last \


# Execute the local evaluation
#echo "----- LOCAL EVALUATION -----"
if [ ! -n "${AICROWD_IS_GRADING+set}" ]; then
    python ${NDC_ROOT}/local_evaluation.py
fi
