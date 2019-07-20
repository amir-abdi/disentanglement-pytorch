#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--aicrowd_challenge=true \
--name=$NAME \
--alg=BetaVAE \
--vae_loss=AnnealedCapacity \
--vae_type=FactorVAE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA \
--dset_name=$AICROWD_DATASET_NAME \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=true \
--w_kld=0.5 \
--lr_G=0.0008 \
--max_iter=80000 \
--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/neurips_mpi3d_toy_BetaVAE_FactorVAE/last \




