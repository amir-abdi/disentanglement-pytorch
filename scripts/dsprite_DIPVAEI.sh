#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--loss_terms=DIPVAEI \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=dsprites_full \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=false \
--w_kld=1 \
--lr_G=0.0005 \




