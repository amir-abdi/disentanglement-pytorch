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
--vae_type=BetaTCVAE \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=true \
--w_kld=2.0 \
--w_tc_analytical=2.0 \
--lr_G=0.005 \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_args mode=min factor=0.91 patience=1 min_lr=0.00005 \
--max_iter=30000 \
--ckpt_load_iter=false \
--ckpt_load=./saved_models/celebA_ae_gaussian/saved \
--ckpt_load_optim=false \
--batch_size=256 \

#--ckpt_load=./checkpoints/mpi3dtoy_betatc-vae/last \
#--ckpt_load_iter=false \


#--ckpt_load=checkpoints/mpi3dtoy_betatc-vae/last \

#--ckpt_load=./saved_models/mpi3dtoy_ae_gaussian/saved2 \










