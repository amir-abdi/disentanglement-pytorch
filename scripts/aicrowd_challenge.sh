#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--aicrowd_challenge=true \
--name=$NAME \
--alg=BetaVAE \
--annealed_capacity=true \
--loss_terms=BetaTCVAE \
--traverse_z=true \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=false \
--w_kld=1.0 \
--w_tc=2.0 \
--lr_G=0.001 \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_args mode=min factor=0.95 patience=1 min_lr=0.00005 \
--max_iter=90000 \
--iterations_c=2000 \
--evaluate_metric mig sap_score irs factor_vae_metric dci \
