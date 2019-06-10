[![CircleCI](https://circleci.com/gh/amir-abdi/disentanglement-pytorch.svg?style=svg&circle-token=40d47183b78c6f1959ff584259c89ac7d49e36b0)](https://circleci.com/gh/amir-abdi/disentanglement-pytorch)

# disentanglement-pytorch
Disentanglement Framework for Pytorch with most well-known algorithms of the realm


### Setup
To smoothly run the demo scripts, set the `DATASETS` environment variable 
to the directory holding all the datasets. 
Please don't change the original structure of the datasets as the
`common.dataset.get_dataloader()` method checks for the original 
names.