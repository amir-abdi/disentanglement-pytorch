#! /bin/sh

FILENAME=$(basename $0)
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=AE \
--dset_dir=/home/amirabdi/data/Datasets/ \
--dset_name=dsprites \
--traverse_z=true \
--encoder_name=SimpleEncoder64 \
--decoder_name=SimpleDecoder64 \
--num_channels=1 \
--z_dim=8 \
--use_wandb=false \
--batch_size=32 \





