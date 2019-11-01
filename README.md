[![CircleCI](https://circleci.com/gh/amir-abdi/disentanglement-pytorch.svg?style=svg&circle-token=40d47183b78c6f1959ff584259c89ac7d49e36b0)](https://circleci.com/gh/amir-abdi/disentanglement-pytorch)

# disentanglement-pytorch
Pytorch Implementation of **Disentanglement** algorithms for Variational Autoencoders. This library was developed as our little  contribution to the ***[Disentanglement Challenge of NeurIPS 2019](https://aicrowd.com/challenges/neurips-2019-disentanglement-challenge)***.

The following algorithms are implemented:
- VAE
- β-VAE ([Understanding disentangling in β-VAE](https://arxiv.org/pdf/1804.03599.pdf))
- Info-VAE ([InfoVAE: Information Maximizing Variational Autoencoders](https://arxiv.org/abs/1706.02262))
- Beta-TCVAE ([Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942))
- DIP-VAE I & II ([Variational Inference of Disentangled Latent Concepts from Unlabeled Observations ](https://openreview.net/forum?id=H1kG7GZAW))
- Factor-VAE ([Disentangling by Factorising](https://arxiv.org/pdf/1802.05983.pdf))
- CVAE ([Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models.pdf))
- IFCVAE ([Adversarial Information Factorization](https://arxiv.org/pdf/1711.05175.pdf))

We are open to suggestions and contributions.


### Requirements and Installation

Install the requirements: `pip install -r requirements.txt` \
Or build conda environment: `conda env create -f environment.yml`

The library visualizes the ***reconstructed images*** and the ***traversed latent spaces*** and saves them as static frames as well as animated GIFs. It also extensively uses the [Weights & Biases](https://www.wandb.com/) toolkit to log the training (loss, metrics, misc, etc.) and the visualizations.

### Training

    python main.py [[--ARG ARG_VALUE] ...]

or, try one of the bash files in the `scripts/` directory:

    bash scripts/SCRIPT_NAME
    
#### Evaluate
To evaluate the learned disentangled representation, set the `--evaluate_metric` 
flag to a subset of the following available metrics: 
*mig, sap_score, irs, factor_vae_metric, dci* (see `scripts/aicrowd_challenge`).

### Data Setup
To run the scripts:

1- Set the `-dset_dir` flag or the `$DISENTANGLEMENT_LIB_DATA` environment variable to the directory 
holding all the datasets (the former is given priority). 

2- Set the `dset_name` flag or the `$DATASET_NAME` environment variable to the name of the dataset (the former is given priority).
The supported datasets are: 
*[celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html),
[dsprites](https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz)
(and the Deppmind's variants: color, noisy, scream, introduced [here](https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/named_data.py)),
[smallnorb](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/), 
[cars3d](http://www.scottreed.info/files/nips2015-analogy-data.tar.gz), 
[mpi3d_toy](https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_toy_64x_ordered_without_heldout_factors.npz), and 
[mpi3d_realistic](https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_realistic_64x_ordered_without_heldout_factors.npz).  

<!--- [shapes3d](https://storage.cloud.google.com/3d-shapes/3dshapes.h5)*.-->
 
Currently, there are two dataloaders in place: 
- One handles labels for semi-supervised and conditional (class-aware) training (*e.g.* CVAE, IFCVAE) , 
but only supports the `celebA` and `dsprites_full` datasets for now. 
- The other leverages Google's implementations of [disentanglement_lib](https://github.com/google-research/disentanglement_lib),
and is based on the starter kit of the 
[Disentanglement Challenge of NeurIPS 2019](https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/utils_pytorch.py),
hosted by [AIcrowd](http://aicrowd.com).


Check some of the bash scripts in the `scripts/` folder for possibilities.

