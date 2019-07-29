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
--vae_type=BetaTCVAE \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=true \
--w_kld=1.0 \
--w_tc_analytical=1.0 \
--lr_G=0.006 \
--lr_scheduler=StepLR \
--lr_scheduler_args step_size=1 gamma=0.95 \
--max_epoch=60 \
--iterations_c=2000 \
--ckpt_load=./checkpoints/mpi3dtoy_betatc-vae/last \
--ckpt_load_iter=false \
--ckpt_load=./saved_models/saved_models/mpi3dtoy_ae_gaussian/saved \
--ckpt_load_optim=false \
--evaluate_metric mig sap_score irs \

#--ckpt_load=./saved_models/mpi3dtoy_betatc/saved \

#--ckpt_load=./checkpoints/mpi3dtoy_betatc-vae/last \
#--ckpt_load_iter=false \


#--ckpt_load=checkpoints/mpi3dtoy_betatc-vae/last \

#--ckpt_load=./saved_models/mpi3dtoy_ae_gaussian/saved2 \







