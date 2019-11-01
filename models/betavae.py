from models.vae import VAE


class BetaVAE(VAE):
    """
    β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
    by Higgins et al.
    https://openreview.net/pdf?id=Sy2fzU9gl

    Understanding disentangling in β-VAE
    by Burgess et al.
    https://arxiv.org/pdf/1804.03599.pdf
    """

    def __init__(self, args):
        super().__init__(args)
