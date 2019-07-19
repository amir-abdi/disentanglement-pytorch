#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--vae_loss=AnnealedCapacity \
--dset_dir=/home/amirabdi/data/Datasets/ \
--dset_name=dsprites \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=8 \
--use_wandb=true \
--w_recon=10000 \
--w_kld=2 \
--max_c=25.0 \
--iterations_c=100000 \
--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/capacity_test/last \
--wandb_resume_id=inyofp8y \

