#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--controlled_capacity_increase=true \
--loss_terms=BetaTCVAE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA \
--dset_name=dsprites_full \
--traverse_z=true \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=20 \
--use_wandb=false \
--w_kld=1.0 \
--w_tc=2.0 \
--lr_G=0.0001 \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_args mode=min factor=0.8 patience=0 min_lr=0.000001 \
--max_iter=20000 \
--iterations_c=2000 \





