#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=VAE \
--vae_loss=Basic \
--vae_type=Basic \
--dset_dir=/home/amirabdi/data/Datasets/ \
--dset_name=celebA \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=20 \
--use_wandb=true \
--w_kld=1.0 \



