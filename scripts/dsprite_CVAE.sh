#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=CVAE \
--controlled_capacity_increase=true \
--dset_dir=/home/emanuele/disentanglement_lib/  \
--dset_name=dsprites_full \
--traverse_z=true \
--traverse_c=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--label_tiler=MultiTo2DChannel \
--z_dim=8 \
--w_kld=5 \
--lr_G=0.0004 \
--include_labels 1 10 \
--use_wandb=false




