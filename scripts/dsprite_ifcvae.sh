#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=IFCVAE \
--vae_loss=AnnealedCapacity \
--dset_dir=/home/amirabdi/data/Datasets/ \
--dset_name=dsprites \
--traverse_z=true \
--traverse_c=true \
--encoder SimpleGaussianEncoder64 SimpleEncoder64 \
--decoder=SimpleDecoder64 \
--discriminator=SimpleDiscriminator \
--label_tiler=MultiTo2DChannel \
--z_dim=8 \
--w_kld=5 \
--w_le=1 \
--w_aux=20 \
--w_tc=1 \
--include_labels 1 \
--use_wandb=true \
--ckpt_load=/home/amirabdi/disentanglement-pytorch/checkpoints/dsprite_ifcvae/before_softmax \

#--wandb_resume_id=s2hc9d6i \

#--verbose=10 \
#--all_iter=1 \







