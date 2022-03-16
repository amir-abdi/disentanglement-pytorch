import imp
from .ae import AE
from .vae import VAE
from .betavae import BetaVAE
from .cvae import CVAE
from .ifcvae import IFCVAE
from .concept_vae import ConceptVAE
from .grayvae_standard import GrayVAE_Standard
from .grayvae_join import GrayVAE_Join
from .cbm_seq import CBM_Seq
from .cbm_join import CBM_Join

# TODO: add author and license info to all files.
# TODO: 3 different divergences in the InfoVAE paper https://arxiv.org/pdf/1706.02262.pdf
# TODO: evaluation metrics
# TODO: Add Adversarial Autoencoders https://arxiv.org/pdf/1511.05644.pdf
# TODO: A version of CVAE where independence between C and Z is enforced
# TODO: Add PixelCNN and PixelCNN++ and PixelVAE
# TODO: Add VQ-VAE (discrete encodings) and VQ-VAE2 --> I guess Version 2 has pixelCNN
# TODO: SCGAN_Disentangled_Representation_Learning_by_Addi
