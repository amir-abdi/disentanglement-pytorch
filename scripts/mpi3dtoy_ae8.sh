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
--vae_type=Vanilla \
--dset_dir=$DISENTANGLEMENT_LIB_DATA \
--dset_name=$AICROWD_DATASET_NAME \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=8 \
--use_wandb=true \
--w_kld=0.0 \
--lr_G=0.0026 \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_args mode=min factor=0.95 patience=2 min_lr=0.000001 \
--max_iter=220000 \
--ckpt_load=checkpoints/mpi3dtoy_ae8/last \
--ckpt_load_optim=false \

#--ckpt_load_iternum=false \



#--ckpt_load_iternum=false \







