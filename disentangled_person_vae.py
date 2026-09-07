import torch
import torch.nn as nn

from common.ops import reparametrize


# ============================================================
# GRADIENT REVERSAL
# ============================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer.

    Forward pass:
        behaves like an identity function.

            y = x

    Backward pass:
        reverses the gradient that flows into x.

            dL/dx = -lambda * dL/dy

    This allows the utility identity adversary to learn how
    to predict identity, while simultaneously encouraging
    the encoder to REMOVE identity information from
    mu_utility.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        grl_lambda
    ):
        ctx.grl_lambda = grl_lambda

        # Forward pass does nothing to the representation.
        return x.view_as(x)


    @staticmethod
    def backward(
        ctx,
        grad_output
    ):
        # Reverse the gradient flowing toward the encoder.
        grad_input = (
            -ctx.grl_lambda
            *
            grad_output
        )

        # No gradient is required for grl_lambda itself.
        return (
            grad_input,
            None
        )


class GradientReversalLayer(nn.Module):
    """
    nn.Module wrapper around GradientReversalFunction.
    """

    def __init__(
        self,
        grl_lambda=1.0
    ):
        super().__init__()

        self.grl_lambda = grl_lambda


    def forward(
        self,
        x,
        grl_lambda=None
    ):
        # Allow the training script to override lambda
        # dynamically if desired.

        if grl_lambda is None:
            grl_lambda = self.grl_lambda

        return GradientReversalFunction.apply(
            x,
            grl_lambda
        )


# ============================================================
# DISENTANGLED PERSON VAE
# ============================================================

class DisentangledPersonVAE(nn.Module):
    """
    Phase 2D.4 Disentangled Person VAE.

    Latent representation:

        z = [z_private, z_utility]

    ----------------------------------------------------------
    PRIVATE BRANCH
    ----------------------------------------------------------

    mu_private:
        encouraged to contain identity-sensitive information.

        mu_private
            ↓
        private identity classifier
            ↓
        predict PID correctly

    ----------------------------------------------------------
    UTILITY BRANCH
    ----------------------------------------------------------

    mu_utility:
        encouraged to contain less absolute identity
        information.

        mu_utility
            ↓
        Gradient Reversal Layer
            ↓
        utility identity adversary
            ↓
        predict PID

    The adversary itself learns to classify identity normally.

    However, because of the GRL, the encoder receives the
    opposite gradient and therefore learns to make identity
    classification from mu_utility increasingly difficult.

    ----------------------------------------------------------
    RECONSTRUCTION
    ----------------------------------------------------------

    Sampled latent z is still used for reconstruction:

        z_private + z_utility
            ↓
        decoder
            ↓
        reconstructed person

    Deterministic means are used for identity-related
    objectives:

        mu_private -> private identity classifier
        mu_utility -> adversarial identity classifier
    """

    def __init__(
        self,
        encoder,
        decoder,
        private_dim=64,
        utility_dim=64,
        num_identities=751,
        grl_lambda=1.0
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.private_dim = private_dim
        self.utility_dim = utility_dim

        self.latent_dim = (
            private_dim
            +
            utility_dim
        )

        self.num_identities = num_identities

        self.grl_lambda = grl_lambda


        # ====================================================
        # PRIVATE IDENTITY CLASSIFIER
        # ====================================================

        # mu_private:
        #
        # [B, 64]
        #     ↓
        # [B, 751]

        self.private_classifier = nn.Linear(
            private_dim,
            num_identities
        )


        # ====================================================
        # GRADIENT REVERSAL LAYER
        # ====================================================

        self.gradient_reversal = (
            GradientReversalLayer(
                grl_lambda=grl_lambda
            )
        )


        # ====================================================
        # UTILITY IDENTITY ADVERSARY
        # ====================================================

        # We use a slightly stronger classifier than a single
        # linear layer so that the encoder cannot trivially hide
        # identity information that is still easily recoverable
        # through a nonlinear mapping.
        #
        # mu_utility:
        #
        # [B, 64]
        #     ↓
        # [B, 128]
        #     ↓
        # [B, 751]

        self.utility_identity_adversary = nn.Sequential(
            nn.Linear(
                utility_dim,
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Linear(
                128,
                num_identities
            )
        )


    # ========================================================
    # LATENT SPLIT
    # ========================================================

    def split_latent(
        self,
        latent
    ):

        private = latent[
            :,
            :self.private_dim
        ]

        utility = latent[
            :,
            self.private_dim:
        ]

        return (
            private,
            utility
        )


    # ========================================================
    # LATENT COMBINATION
    # ========================================================

    def combine_latent(
        self,
        z_private,
        z_utility
    ):

        z = torch.cat(
            [
                z_private,
                z_utility
            ],
            dim=1
        )

        return z


    # ========================================================
    # ENCODER
    # ========================================================

    def encode(
        self,
        x
    ):

        # ----------------------------------------------------
        # Encoder predicts q(z|x)
        # ----------------------------------------------------

        mu, logvar = self.encoder(
            x
        )


        # ----------------------------------------------------
        # Deterministic latent means
        #
        # Used for identity-related objectives.
        # ----------------------------------------------------

        (
            mu_private,
            mu_utility
        ) = self.split_latent(
            mu
        )


        # ----------------------------------------------------
        # Stochastic latent representation
        #
        # Used for reconstruction.
        # ----------------------------------------------------

        z = reparametrize(
            mu,
            logvar
        )


        (
            z_private,
            z_utility
        ) = self.split_latent(
            z
        )


        return {
            "mu":
                mu,

            "logvar":
                logvar,

            "mu_private":
                mu_private,

            "mu_utility":
                mu_utility,

            "z_private":
                z_private,

            "z_utility":
                z_utility
        }


    # ========================================================
    # DECODER
    # ========================================================

    def decode(
        self,
        z_private,
        z_utility
    ):

        z = self.combine_latent(
            z_private,
            z_utility
        )


        logits = self.decoder(
            z
        )


        reconstruction = torch.sigmoid(
            logits
        )


        return reconstruction


    # ========================================================
    # PRIVATE IDENTITY CLASSIFIER
    # ========================================================

    def classify_private(
        self,
        mu_private
    ):

        private_identity_logits = (
            self.private_classifier(
                mu_private
            )
        )


        return private_identity_logits


    # ========================================================
    # UTILITY IDENTITY ADVERSARY
    # ========================================================

    def classify_utility_adversarial(
        self,
        mu_utility,
        grl_lambda=None
    ):

        # ----------------------------------------------------
        # Forward:
        #
        # mu_utility is unchanged.
        #
        # Backward:
        #
        # gradient flowing into the encoder is reversed.
        # ----------------------------------------------------

        reversed_utility = (
            self.gradient_reversal(
                mu_utility,
                grl_lambda=grl_lambda
            )
        )


        utility_identity_logits = (
            self.utility_identity_adversary(
                reversed_utility
            )
        )


        return utility_identity_logits


    # ========================================================
    # FULL FORWARD PASS
    # ========================================================

    def forward(
        self,
        x,
        grl_lambda=None
    ):

        # ----------------------------------------------------
        # Encode image
        # ----------------------------------------------------

        latent_outputs = self.encode(
            x
        )


        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        reconstruction = self.decode(
            latent_outputs[
                "z_private"
            ],
            latent_outputs[
                "z_utility"
            ]
        )


        # ----------------------------------------------------
        # Private identity prediction
        #
        # Encoder is encouraged to make this prediction easy.
        # ----------------------------------------------------

        private_identity_logits = (
            self.classify_private(
                latent_outputs[
                    "mu_private"
                ]
            )
        )


        # ----------------------------------------------------
        # Utility identity adversary
        #
        # Adversary:
        #   tries to predict PID.
        #
        # Encoder:
        #   receives reversed gradient and therefore tries to
        #   make PID difficult to predict from mu_utility.
        # ----------------------------------------------------

        utility_identity_logits = (
            self.classify_utility_adversarial(
                latent_outputs[
                    "mu_utility"
                ],
                grl_lambda=grl_lambda
            )
        )


        # ----------------------------------------------------
        # Return all useful representations
        # ----------------------------------------------------

        return {
            "reconstruction":
                reconstruction,

            "mu":
                latent_outputs[
                    "mu"
                ],

            "logvar":
                latent_outputs[
                    "logvar"
                ],

            "z_private":
                latent_outputs[
                    "z_private"
                ],

            "z_utility":
                latent_outputs[
                    "z_utility"
                ],

            "mu_private":
                latent_outputs[
                    "mu_private"
                ],

            "mu_utility":
                latent_outputs[
                    "mu_utility"
                ],

            "private_identity_logits":
                private_identity_logits,

            "utility_identity_logits":
                utility_identity_logits
        }