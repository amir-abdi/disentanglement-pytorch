[![CircleCI](https://circleci.com/gh/amir-abdi/disentanglement-pytorch.svg?style=svg&circle-token=40d47183b78c6f1959ff584259c89ac7d49e36b0)](https://circleci.com/gh/amir-abdi/disentanglement-pytorch)

# disentanglement-pytorch
Pytorch Implementation of Disentanglement Algorithms for Variational Autoencoders 


### Installation

Install the requirements: `pip install -r requirements.txt`
Or build conda environment: `conda env create -f environment.yml`


### Training

To run the models:

    python main.py [[--ARG ARG_VALUE] ...]

There are some ready-to-run scripts in the `scripts` folder. Try them with:

    bash scripts/SCRIPT_NAME
    

### Data Setup
To run the scripts:
- set the `$DATASETS` environment variable 
to the directory holding all the datasets, and
- set the  `$AICROWD_DATASET_NAME` environment variable to
the name of the dataset 
(borrowed from [here](https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/utils_pytorch.py)
which leverages the implementations of [disentanglement_lib](https://github.com/google-research/disentanglement_lib)).

The loader for the following datasets are implemented:
'dsprites_full', 'dsprites_noshape', 'color_dsprites', 
'noisy_dsprites', 'scream_dsprites', 'smallnorb', 
'cars3d', 'mpi3d_toy', 'mpi3d_realistic', 
'mpi3d_real', 'shapes3d'.  

The alternative is to set the `--dset_dir` and `--dset_name` flags to 
where the dataset is stored. Only the 'CelebA' and
'dsprites' datasets are included in this approach. 
