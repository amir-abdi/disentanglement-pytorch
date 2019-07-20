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
--vae_type=FactorVAE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA \
--dset_name=$AICROWD_DATASET_NAME \
--traverse_z=true \
--encoder=DeepGaussianLinear \
--decoder=DeepLinear \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=true \
--w_kld=0.5 \
--lr_G=0.005 \
--max_iter=60000 \
--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/neurips_linear_mpi3d_toy_BetaVAE_FactorVAE/last \




