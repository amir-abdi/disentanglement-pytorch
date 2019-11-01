#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=IFCVAE \
--vae_loss=AnnealedCapacity \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=dsprites_full \
--traverse_z=true \
--traverse_c=true \
--encoder SimpleGaussianConv64 SimpleConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--label_tiler=MultiTo2DChannel \
--z_dim=8 \
--w_kld=5 \
--w_le=1 \
--w_aux=20 \
--w_tc_empirical=1 \
--include_labels 1 \
--use_wandb=false \
