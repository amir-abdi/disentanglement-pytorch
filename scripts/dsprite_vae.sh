#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--vae_loss=Basic \
--dset_dir=$DATASETS \
--dset_name=dsprites \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=8 \
--w_kld=1 \
--use_wandb=false




