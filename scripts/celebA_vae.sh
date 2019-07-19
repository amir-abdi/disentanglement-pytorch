#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=VAE \
--vae_loss=Basic \
--dset_dir=/home/amirabdi/data/Datasets/ \
--dset_name=celebA \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=8 \
--use_wandb=false \
--w_recon=10000 \
--w_kld=1 \



