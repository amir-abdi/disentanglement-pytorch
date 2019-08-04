#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--aicrowd_challenge=true \
--name=$NAME \
--alg=VAE \
--vae_loss=Basic \
--vae_type=DIPVAE \
--dip_type=ii \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--traverse_z=true \
--z_dim=20 \
--use_wandb=true \
--w_kld=1 \
--lr_G=0.003 \
--max_iter=20000 \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_args mode=min factor=0.98 patience=2 min_lr=0.00001 \
--ckpt_load_iternum=false \
--ckpt_load=checkpoints/mpi3dtoy_dipII-vae2/last \
--ckpt_load_optim=false
