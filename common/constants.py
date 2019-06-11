# Strings
LOSS = 'loss'
ACCURACY = 'acc'
ITERATION = 'iteration'
WANDB_NAME = 'disentanglement'
INPUT_IMAGE = 'input_image'
RECON_IMAGE = 'recon_image'
RECON = 'recon'
FIXED = 'fixed'
SQUARE = 'square'
ELLIPSE = 'ellipse'
HEART = 'heart'
TRAVERSE = 'traverse'
RANDOM = 'random'
TEMP = 'tmp'
GIF = 'gif'
JPG = 'jpg'
FACTORVAE = 'FactorVAE'
VAE = 'vae'

# Algorithms
ALGS = ('AE', 'VAE', 'BetaVAE', 'CVAE', 'IFCVAE')
VAE_LOSS = ('Basic', 'AnnealedCapacity')
VAE_TYPE = ('Vanilla', FACTORVAE,)
DATASETS = ('celebA', 'dsprites')

# Architectures
DISCRIMINATORS = ('SimpleDiscriminator', 'SimpleDiscriminatorConv64')
TILERS = ('MultiTo2DChannel',)
DECODERS = ('SimpleDecoder64',)
ENCODERS = ('SimpleEncoder64', 'SimpleGaussianEncoder64',)

