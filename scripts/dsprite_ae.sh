#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=AE \
--dset_dir=$DATASETS \
--dset_name=dsprites \
--encoder=SimpleEncoder64 \
--decoder=SimpleDecoder64 \
--z_dim=8 \
--w_recon=10000 \
--use_wandb=false \
--ckpt_load=./saved_models/dsprite_ae/last





