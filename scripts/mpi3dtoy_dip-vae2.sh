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
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=True \
--w_kld=1 \
--lr_G=0.0008 \
--max_iter=60000 \
--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/mpi3dtoy_dip-vae2/saved40000

#--ckpt_load=./saved_models/mpi3dtoy_ae_gaussian/saved2 \
#--ckpt_load_iter=false \





