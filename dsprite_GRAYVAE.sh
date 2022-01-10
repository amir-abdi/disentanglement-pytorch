#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=GRAYVAE \
--controlled_capacity_increase=true \
--dset_dir=/home/emanuele/disentanglement_lib/  \
--dset_name=dsprites_full \
--traverse_z=true \
--traverse_c=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--label_tiler=MultiTo2DChannel \
--z_dim=8 \
--lr_G=0.0004 \
--include_labels 1 2 3 4 5 \
--use_wandb=false \
--max_iter=5000 \
--w_recon=0.1 \
--evaluation_metric mig \
--evaluate_iter=8 \
--batch_size=10
