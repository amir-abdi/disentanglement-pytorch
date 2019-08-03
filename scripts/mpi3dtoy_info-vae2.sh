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
--vae_type=InfoVAE \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=true \
--w_kld=1.0 \
--w_infovae=1000 \
--lr_G=0.0005 \
--max_iter=100000 \
--iterations_c=2000 \
--evaluate_metric mig sap_score irs factor_vae_metric \
--batch_size=64 \
--ckpt_load_iter=false \
--ckpt_load=./saved_models/mpi3dtoy_betatc/saved \
--ckpt_load_optim=false \

#--ckpt_load_iter=false \


#--ckpt_load=checkpoints/mpi3dtoy_betatc-vae/last \

#--ckpt_load=./saved_models/mpi3dtoy_ae_gaussian/saved2 \










