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
--vae_type=Basic \
--dset_dir=$DISENTANGLEMENT_LIB_DATA \
--dset_name=$AICROWD_DATASET_NAME \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=true \
--w_kld=1.0 \
--w_tc_empirical=0.0 \
--lr_G=0.005 \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_args mode=min factor=0.9 patience=1 min_lr=0.000001 \
--max_iter=500000 \
--iterations_c=60000 \
--batch_size=128 \
--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/mpi3dtoy_ae/last \






