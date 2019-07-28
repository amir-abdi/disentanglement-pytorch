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
--vae_type FactorVAE BetaTCVAE \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=true \
--w_kld=1.0 \
--w_tc_analytical=5.0 \
--w_tc_empirical=1.0 \
--lr_G=0.0038 \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_args mode=min factor=0.92 patience=3 min_lr=0.000001 \
--max_iter=20000 \
--iterations_c=2000 \
--ckpt_load_iternum=false \
--ckpt_load=./saved_models/celebA_ae_gaussian/saved \
--ckpt_load_optim=false \
--evaluate_metric mig sap_score irs \
--batch_size=256 \

#--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/mpi3dtoy_betatc-vae-saved4/last \
#--wandb_resume_id=r9r2nbvn \


#--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/mpi3dtoy_betatc-vae-saved2/last \



#--ckpt_load=./checkpoints/mpi3dtoy_betatc-vae/last \
#--ckpt_load_iter=false \


#--ckpt_load=checkpoints/mpi3dtoy_betatc-vae/last \

#--ckpt_load=./saved_models/mpi3dtoy_ae_gaussian/saved2 \










