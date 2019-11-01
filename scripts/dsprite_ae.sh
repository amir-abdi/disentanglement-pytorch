#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=AE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=dsprites_full \
--encoder=SimpleConv64 \
--decoder=SimpleConv64 \
--z_dim=8 \
--w_recon=10000 \
--use_wandb=false \





