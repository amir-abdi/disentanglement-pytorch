#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--aicrowd_challenge=true \
--name=$NAME \
--alg=BetaVAE \
--vae_loss=Basic \
--vae_type=DIPVAE \
--dip_type=i \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=True \
--w_kld=1 \
--lr_G=0.0005 \
--max_iter=100000
#--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/dsprite_dip-vae/last \




