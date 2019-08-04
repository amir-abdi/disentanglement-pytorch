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
--aicrowd_challenge=true \
--dset_dir=$DISENTANGLEMENT_LIB_DATA \
--dset_name=$AICROWD_DATASET_NAME \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=20 \
--use_wandb=true \
--w_kld=1.0 \
--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/celebA_vae2/last



