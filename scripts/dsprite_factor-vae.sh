#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
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
--z_dim=8 \
--use_wandb=true \
--w_kld=1 \
--lr_G=0.0002 \
--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/dsprite_vae_factor/last \




