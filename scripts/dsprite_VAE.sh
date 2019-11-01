#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=dsprites_full \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=8 \
--w_kld=1 \
--use_wandb=false




