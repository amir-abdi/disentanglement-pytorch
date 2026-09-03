import os
import glob
import csv
import re
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import matplotlib.pyplot as plt

# Custom pedestrian encoder and decoder
from architectures.encoders.person_conv128x64 import (
    PersonGaussianConv128x64
)

from architectures.decoders.person_conv128x64 import (
    PersonConv128x64
)

# VAE KL divergence
from common.ops import kl_divergence_mu0_var1

# Disentangled VAE with Gradient Reversal Layer
from disentangled_person_vae import DisentangledPersonVAE


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


MARKET1501_TRAIN_DIR = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15/"
    "bounding_box_train"
)


# ============================================================
# PRETRAINED PHASE 2D.2b CHECKPOINT
# ============================================================

# IMPORTANT:
#
# We restart from the GOOD model before the failed
# adversarial experiment.
#
# This checkpoint previously gave:
#
# Private probe  = 99.85%
# Utility probe  = 5.14%

PRETRAINED_PRIVATE_CHECKPOINT = (
    "outputs/market1501_disentangled_vae_private_v2/"
    "checkpoints/private_v2_epoch_10.pth"
)


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64

NUM_CHANNELS = 3

LATENT_DIM = 128

PRIVATE_DIM = 64
UTILITY_DIM = 64

assert PRIVATE_DIM + UTILITY_DIM == LATENT_DIM

NUM_IDENTITIES = 751


# ============================================================
# 2. LOSS WEIGHTS
# ============================================================

# KL regularisation
BETA_KL = 0.001


# Private identity classification
LAMBDA_PRIVATE = 1.0


# Utility adversarial loss
LAMBDA_ADVERSARY = 1.0


# ------------------------------------------------------------
# Maximum GRL strength
#
# We do NOT immediately apply 0.1.
#
# During adversarial fine-tuning, GRL will gradually increase:
#
# 0.0 -> ... -> 0.1
# ------------------------------------------------------------

GRL_MAX = 0.1


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 8

NUM_WORKERS = 4


# ------------------------------------------------------------
# STAGE A:
# Train utility adversary before asking encoder to fool it.
# ------------------------------------------------------------

ADVERSARY_WARMUP_EPOCHS = 5


# ------------------------------------------------------------
# STAGE B:
# Alternating adversarial fine-tuning.
# ------------------------------------------------------------

NUM_EPOCHS = 5


# ------------------------------------------------------------
# Main model:
#
# encoder
# decoder
# private classifier
# ------------------------------------------------------------

MAIN_LEARNING_RATE = 1e-4


# ------------------------------------------------------------
# Adversary receives a larger learning rate.
#
# This matches the order of magnitude that worked well
# in the independent identity probe.
# ------------------------------------------------------------

ADVERSARY_LEARNING_RATE = 1e-3


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "market1501_disentangled_vae_adversarial_v2"
)


CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)


RECON_DIR = os.path.join(
    OUTPUT_DIR,
    "reconstructions"
)


os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


os.makedirs(
    RECON_DIR,
    exist_ok=True
)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 78)

print(
    "Phase 2D.4b - Strong Utility Adversarial Fine-Tuning"
)

print("=" * 78)


print(
    "Device:",
    device
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print()

print(
    f"Latent dimension: {LATENT_DIM}"
)

print(
    f"Private dimensions: {PRIVATE_DIM}"
)

print(
    f"Utility dimensions: {UTILITY_DIM}"
)

print(
    f"Training identities: {NUM_IDENTITIES}"
)

print()

print(
    f"BETA_KL: {BETA_KL}"
)

print(
    f"LAMBDA_PRIVATE: {LAMBDA_PRIVATE}"
)

print(
    f"LAMBDA_ADVERSARY: {LAMBDA_ADVERSARY}"
)

print(
    f"GRL_MAX: {GRL_MAX}"
)

print()

print(
    f"Adversary warm-up epochs: "
    f"{ADVERSARY_WARMUP_EPOCHS}"
)

print(
    f"Adversarial fine-tuning epochs: "
    f"{NUM_EPOCHS}"
)

print(
    f"Main learning rate: "
    f"{MAIN_LEARNING_RATE}"
)

print(
    f"Adversary learning rate: "
    f"{ADVERSARY_LEARNING_RATE}"
)

print()


# ============================================================
# 5. MARKET-1501 DATASET
# ============================================================

class Market1501VAEDataset(Dataset):

    def __init__(
        self,
        image_dir,
        transform=None
    ):

        self.image_dir = image_dir
        self.transform = transform


        all_image_paths = sorted(
            glob.glob(
                os.path.join(
                    image_dir,
                    "*.jpg"
                )
            )
        )


        if len(
            all_image_paths
        ) == 0:

            raise RuntimeError(
                f"No JPG images found in:\n"
                f"{image_dir}"
            )


        pattern = re.compile(
            r"([-\d]+)_c(\d)"
        )


        self.samples = []

        original_pids = []


        for image_path in all_image_paths:

            filename = os.path.basename(
                image_path
            )


            match = pattern.search(
                filename
            )


            if match is None:
                continue


            pid = int(
                match.group(1)
            )


            camid = int(
                match.group(2)
            )


            if pid == -1:
                continue


            self.samples.append(
                {
                    "path":
                        image_path,

                    "original_pid":
                        pid,

                    "camid":
                        camid
                }
            )


            original_pids.append(
                pid
            )


        unique_pids = sorted(
            set(
                original_pids
            )
        )


        self.pid2label = {
            pid: label
            for label, pid
            in enumerate(
                unique_pids
            )
        }


        if len(
            self.pid2label
        ) != NUM_IDENTITIES:

            raise RuntimeError(
                f"Expected {NUM_IDENTITIES} identities, "
                f"but found {len(self.pid2label)}."
            )


        print(
            f"Found {len(self.samples)} "
            f"Market-1501 training images"
        )


        print(
            f"Found {len(self.pid2label)} "
            f"training identities"
        )


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        sample = self.samples[
            index
        ]


        image = Image.open(
            sample[
                "path"
            ]
        ).convert(
            "RGB"
        )


        if self.transform is not None:

            image = self.transform(
                image
            )


        pid = self.pid2label[
            sample[
                "original_pid"
            ]
        ]


        return (
            image,
            pid
        )


# ============================================================
# 6. PREPROCESSING
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_HEIGHT,
                IMAGE_WIDTH
            )
        ),

        transforms.ToTensor()
    ]
)


dataset = Market1501VAEDataset(
    MARKET1501_TRAIN_DIR,
    transform=transform
)


data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 7. FIXED RECONSTRUCTION BATCH
# ============================================================

fixed_samples = [
    dataset[i]
    for i in range(
        min(
            8,
            len(dataset)
        )
    )
]


fixed_images = torch.stack(
    [
        sample[0]
        for sample
        in fixed_samples
    ]
).to(
    device
)


# ============================================================
# 8. BUILD MODEL
# ============================================================

encoder = PersonGaussianConv128x64(
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS,
    image_size=None
).to(
    device
)


decoder = PersonConv128x64(
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS,
    image_size=None
).to(
    device
)


model = DisentangledPersonVAE(
    encoder=encoder,
    decoder=decoder,
    private_dim=PRIVATE_DIM,
    utility_dim=UTILITY_DIM,
    num_identities=NUM_IDENTITIES,
    grl_lambda=GRL_MAX
).to(
    device
)


# ============================================================
# 9. LOAD GOOD PHASE 2D.2b CHECKPOINT
# ============================================================

print(
    "Loading Phase 2D.2b epoch-10 checkpoint..."
)


checkpoint = torch.load(
    PRETRAINED_PRIVATE_CHECKPOINT,
    map_location=device,
    weights_only=False
)


model.encoder.load_state_dict(
    checkpoint[
        "encoder_state_dict"
    ]
)


model.decoder.load_state_dict(
    checkpoint[
        "decoder_state_dict"
    ]
)


model.private_classifier.load_state_dict(
    checkpoint[
        "private_classifier_state_dict"
    ]
)


print(
    "Loaded pretrained encoder."
)

print(
    "Loaded pretrained decoder."
)

print(
    "Loaded pretrained private classifier."
)

print(
    "Utility adversary starts from scratch."
)

print()

print(
    "Checkpoint:"
)

print(
    PRETRAINED_PRIVATE_CHECKPOINT
)

print()


# ============================================================
# 10. HELPER: ENABLE / DISABLE PARAMETERS
# ============================================================

def set_requires_grad(
    module,
    requires_grad
):

    for parameter in module.parameters():

        parameter.requires_grad = (
            requires_grad
        )


# ============================================================
# 11. TWO OPTIMIZERS
# ============================================================

# ------------------------------------------------------------
# MAIN OPTIMIZER
#
# The utility adversary is deliberately NOT included.
# ------------------------------------------------------------

main_optimizer = torch.optim.Adam(
    list(
        model.encoder.parameters()
    )
    +
    list(
        model.decoder.parameters()
    )
    +
    list(
        model.private_classifier.parameters()
    ),
    lr=MAIN_LEARNING_RATE
)


# ------------------------------------------------------------
# ADVERSARY OPTIMIZER
# ------------------------------------------------------------

adversary_optimizer = torch.optim.Adam(
    model.utility_identity_adversary.parameters(),
    lr=ADVERSARY_LEARNING_RATE
)


# ============================================================
# 12. MAIN MODEL LOSS
# ============================================================

def compute_main_losses(
    reconstructed,
    original,
    mu,
    logvar,
    private_identity_logits,
    utility_identity_logits,
    pids
):

    recon_loss = F.binary_cross_entropy(
        reconstructed,
        original,
        reduction="mean"
    )


    kl_loss = (
        kl_divergence_mu0_var1(
            mu,
            logvar
        )
    )


    private_identity_loss = (
        F.cross_entropy(
            private_identity_logits,
            pids
        )
    )


    utility_identity_loss = (
        F.cross_entropy(
            utility_identity_logits,
            pids
        )
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # No minus sign here.
    #
    # GRL already reverses the gradient entering the encoder.
    # --------------------------------------------------------

    total_loss = (
        recon_loss

        +

        BETA_KL
        *
        kl_loss

        +

        LAMBDA_PRIVATE
        *
        private_identity_loss

        +

        LAMBDA_ADVERSARY
        *
        utility_identity_loss
    )


    return (
        total_loss,
        recon_loss,
        kl_loss,
        private_identity_loss,
        utility_identity_loss
    )


# ============================================================
# 13. SAVE RECONSTRUCTIONS
# ============================================================

def save_reconstruction_grid(
    originals,
    reconstructions,
    epoch
):

    number_to_save = min(
        8,
        originals.size(0)
    )


    originals = originals[
        :number_to_save
    ]


    reconstructions = reconstructions[
        :number_to_save
    ]


    comparison = torch.cat(
        [
            originals,
            reconstructions
        ],
        dim=0
    )


    save_path = os.path.join(
        RECON_DIR,
        f"epoch_{epoch:02d}.png"
    )


    save_image(
        comparison,
        save_path,
        nrow=number_to_save
    )


    print(
        f"Saved reconstruction grid to "
        f"{save_path}"
    )


# ============================================================
# 14. STAGE A — ADVERSARY WARM-UP
# ============================================================

print()
print("=" * 78)

print(
    "STAGE A - UTILITY ADVERSARY WARM-UP"
)

print("=" * 78)

print()

print(
    "Encoder is frozen."
)

print(
    "Utility adversary learns to predict PID from mu_utility."
)

print()


# ------------------------------------------------------------
# Freeze everything except utility adversary.
# ------------------------------------------------------------

set_requires_grad(
    model.encoder,
    False
)

set_requires_grad(
    model.decoder,
    False
)

set_requires_grad(
    model.private_classifier,
    False
)

set_requires_grad(
    model.utility_identity_adversary,
    True
)


warmup_history_loss = []

warmup_history_accuracy = []


for epoch in range(
    1,
    ADVERSARY_WARMUP_EPOCHS + 1
):

    # Prevent BatchNorm running statistics in the encoder
    # from changing during adversary warm-up.

    model.encoder.eval()

    model.decoder.eval()

    model.private_classifier.eval()

    model.utility_identity_adversary.train()


    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    for batch_idx, (
        images,
        pids
    ) in enumerate(
        data_loader
    ):

        images = images.to(
            device,
            non_blocking=True
        )


        pids = pids.to(
            device,
            non_blocking=True
        )


        adversary_optimizer.zero_grad()


        # ----------------------------------------------------
        # Extract frozen deterministic utility representation.
        # ----------------------------------------------------

        with torch.no_grad():

            mu, _ = model.encoder(
                images
            )


            _, mu_utility = (
                model.split_latent(
                    mu
                )
            )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # No Gradient Reversal Layer during warm-up.
        #
        # We simply want the adversary to become competent.
        # ----------------------------------------------------

        utility_logits = (
            model.utility_identity_adversary(
                mu_utility.detach()
            )
        )


        adversary_loss = F.cross_entropy(
            utility_logits,
            pids
        )


        adversary_loss.backward()


        adversary_optimizer.step()


        predictions = torch.argmax(
            utility_logits,
            dim=1
        )


        correct = (
            predictions
            ==
            pids
        ).sum().item()


        total_loss += (
            adversary_loss.item()
        )


        total_correct += (
            correct
        )


        total_samples += (
            pids.size(0)
        )


        if batch_idx % 100 == 0:

            batch_accuracy = (
                correct
                /
                pids.size(0)
                *
                100
            )


            print(
                f"Warm-up "
                f"[{epoch:02d}/"
                f"{ADVERSARY_WARMUP_EPOCHS}] "
                f"Batch "
                f"[{batch_idx:04d}/"
                f"{len(data_loader)}] "
                f"Loss: "
                f"{adversary_loss.item():.4f} "
                f"Acc: "
                f"{batch_accuracy:.2f}%"
            )


    average_loss = (
        total_loss
        /
        len(data_loader)
    )


    accuracy = (
        total_correct
        /
        total_samples
        *
        100
    )


    warmup_history_loss.append(
        average_loss
    )


    warmup_history_accuracy.append(
        accuracy
    )


    print()

    print(
        f"Warm-up epoch "
        f"{epoch}/"
        f"{ADVERSARY_WARMUP_EPOCHS}"
    )


    print(
        f"Average adversary loss: "
        f"{average_loss:.6f}"
    )


    print(
        f"Adversary accuracy: "
        f"{accuracy:.2f}%"
    )

    print()


# ============================================================
# 15. SAVE WARM-UP ADVERSARY
# ============================================================

warmup_checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "utility_adversary_after_warmup.pth"
)


torch.save(
    {
        "utility_adversary_state_dict":
            model.utility_identity_adversary.state_dict(),

        "optimizer_state_dict":
            adversary_optimizer.state_dict(),

        "warmup_epochs":
            ADVERSARY_WARMUP_EPOCHS,

        "final_warmup_loss":
            warmup_history_loss[-1],

        "final_warmup_accuracy":
            warmup_history_accuracy[-1]
    },

    warmup_checkpoint_path
)


print(
    f"Saved warmed-up adversary to "
    f"{warmup_checkpoint_path}"
)

print()


# ============================================================
# 16. RESTORE TRAINABLE MAIN MODEL
# ============================================================

set_requires_grad(
    model.encoder,
    True
)

set_requires_grad(
    model.decoder,
    True
)

set_requires_grad(
    model.private_classifier,
    True
)

set_requires_grad(
    model.utility_identity_adversary,
    True
)


# ============================================================
# 17. ADVERSARIAL HISTORY
# ============================================================

history_total = []

history_recon = []

history_kl = []

history_private_loss = []

history_utility_loss = []

history_private_accuracy = []

history_adversary_accuracy = []

history_grl_lambda = []


# ============================================================
# 18. STAGE B — ALTERNATING ADVERSARIAL TRAINING
# ============================================================

print()
print("=" * 78)

print(
    "STAGE B - ALTERNATING ADVERSARIAL TRAINING"
)

print("=" * 78)

print()

print(
    "Each batch now performs:"
)

print(
    "1. Train utility adversary."
)

print(
    "2. Freeze adversary."
)

print(
    "3. Train encoder to fool adversary through GRL."
)

print()


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    total_epoch_loss = 0.0

    total_recon_loss = 0.0

    total_kl_loss = 0.0

    total_private_loss = 0.0

    total_utility_loss = 0.0


    total_private_correct = 0

    total_adversary_correct = 0

    total_samples = 0


    for batch_idx, (
        images,
        pids
    ) in enumerate(
        data_loader
    ):

        images = images.to(
            device,
            non_blocking=True
        )


        pids = pids.to(
            device,
            non_blocking=True
        )


        # ====================================================
        # CURRENT GRL STRENGTH
        # ====================================================

        completed_steps = (
            (epoch - 1)
            *
            len(data_loader)
            +
            batch_idx
        )


        total_steps = (
            NUM_EPOCHS
            *
            len(data_loader)
        )


        progress = (
            completed_steps
            /
            max(
                1,
                total_steps - 1
            )
        )


        current_grl_lambda = (
            GRL_MAX
            *
            progress
        )


        # ====================================================
        # STEP A
        #
        # TRAIN THE UTILITY ADVERSARY
        # ====================================================

        set_requires_grad(
            model.utility_identity_adversary,
            True
        )


        model.encoder.eval()

        model.utility_identity_adversary.train()


        adversary_optimizer.zero_grad()


        # ----------------------------------------------------
        # Encoder is detached during adversary update.
        #
        # Therefore adversary learns against current
        # representation without changing the encoder.
        # ----------------------------------------------------

        with torch.no_grad():

            mu_for_adversary, _ = model.encoder(
                images
            )


            _, mu_utility_detached = (
                model.split_latent(
                    mu_for_adversary
                )
            )


        adversary_logits = (
            model.utility_identity_adversary(
                mu_utility_detached.detach()
            )
        )


        adversary_loss = F.cross_entropy(
            adversary_logits,
            pids
        )


        adversary_loss.backward()


        adversary_optimizer.step()


        adversary_predictions = torch.argmax(
            adversary_logits,
            dim=1
        )


        adversary_correct = (
            adversary_predictions
            ==
            pids
        ).sum().item()


        # ====================================================
        # STEP B
        #
        # TRAIN ENCODER / DECODER / PRIVATE CLASSIFIER
        # ====================================================

        # Freeze adversary parameters.
        #
        # Gradients can still pass THROUGH the adversary into
        # mu_utility, but adversary weights will not change.

        set_requires_grad(
            model.utility_identity_adversary,
            False
        )


        model.encoder.train()

        model.decoder.train()

        model.private_classifier.train()

        # Disable Dropout or other training-time randomness
        # in the adversary during the encoder step.

        model.utility_identity_adversary.eval()


        main_optimizer.zero_grad()


        outputs = model(
            images,
            grl_lambda=current_grl_lambda
        )


        reconstructed = outputs[
            "reconstruction"
        ]


        mu = outputs[
            "mu"
        ]


        logvar = outputs[
            "logvar"
        ]


        mu_private = outputs[
            "mu_private"
        ]


        mu_utility = outputs[
            "mu_utility"
        ]


        private_identity_logits = outputs[
            "private_identity_logits"
        ]


        utility_identity_logits = outputs[
            "utility_identity_logits"
        ]


        (
            loss,
            recon_loss,
            kl_loss,
            private_identity_loss,
            utility_identity_loss
        ) = compute_main_losses(
            reconstructed,
            images,
            mu,
            logvar,
            private_identity_logits,
            utility_identity_logits,
            pids
        )


        loss.backward()


        main_optimizer.step()


        # ----------------------------------------------------
        # Restore adversary trainability for next batch.
        # ----------------------------------------------------

        set_requires_grad(
            model.utility_identity_adversary,
            True
        )


        # ====================================================
        # PRIVATE ACCURACY
        # ====================================================

        private_predictions = torch.argmax(
            private_identity_logits,
            dim=1
        )


        private_correct = (
            private_predictions
            ==
            pids
        ).sum().item()


        batch_private_accuracy = (
            private_correct
            /
            pids.size(0)
            *
            100
        )


        batch_adversary_accuracy = (
            adversary_correct
            /
            pids.size(0)
            *
            100
        )


        # ====================================================
        # STATISTICS
        # ====================================================

        total_epoch_loss += (
            loss.item()
        )


        total_recon_loss += (
            recon_loss.item()
        )


        total_kl_loss += (
            kl_loss.item()
        )


        total_private_loss += (
            private_identity_loss.item()
        )


        total_utility_loss += (
            utility_identity_loss.item()
        )


        total_private_correct += (
            private_correct
        )


        total_adversary_correct += (
            adversary_correct
        )


        total_samples += (
            pids.size(0)
        )


        # ====================================================
        # FIRST BATCH CHECK
        # ====================================================

        if (
            epoch == 1
            and
            batch_idx == 0
        ):

            print(
                "=" * 70
            )

            print(
                "PHASE 2D.4b MODEL CHECK"
            )

            print(
                "=" * 70
            )


            print(
                "Images:",
                tuple(
                    images.shape
                )
            )


            print(
                "mu:",
                tuple(
                    mu.shape
                )
            )


            print(
                "mu_private:",
                tuple(
                    mu_private.shape
                )
            )


            print(
                "mu_utility:",
                tuple(
                    mu_utility.shape
                )
            )


            print(
                "Private identity logits:",
                tuple(
                    private_identity_logits.shape
                )
            )


            print(
                "Utility identity logits:",
                tuple(
                    utility_identity_logits.shape
                )
            )


            print(
                f"Initial GRL lambda: "
                f"{current_grl_lambda:.6f}"
            )


            print(
                "=" * 70
            )

            print()


        # ====================================================
        # PROGRESS
        # ====================================================

        if batch_idx % 50 == 0:

            print(
                f"Epoch "
                f"[{epoch:02d}/{NUM_EPOCHS}] "
                f"Batch "
                f"[{batch_idx:04d}/"
                f"{len(data_loader)}] "
                f"GRL: "
                f"{current_grl_lambda:.4f} "
                f"Total: "
                f"{loss.item():.4f} "
                f"Recon: "
                f"{recon_loss.item():.4f} "
                f"KL: "
                f"{kl_loss.item():.2f} "
                f"Private-ID: "
                f"{private_identity_loss.item():.4f} "
                f"Private-Acc: "
                f"{batch_private_accuracy:.2f}% "
                f"Adv-Loss: "
                f"{adversary_loss.item():.4f} "
                f"Adv-Acc: "
                f"{batch_adversary_accuracy:.2f}%"
            )


    # ========================================================
    # 19. EPOCH RESULTS
    # ========================================================

    average_total = (
        total_epoch_loss
        /
        len(data_loader)
    )


    average_recon = (
        total_recon_loss
        /
        len(data_loader)
    )


    average_kl = (
        total_kl_loss
        /
        len(data_loader)
    )


    average_private_loss = (
        total_private_loss
        /
        len(data_loader)
    )


    average_utility_loss = (
        total_utility_loss
        /
        len(data_loader)
    )


    private_accuracy = (
        total_private_correct
        /
        total_samples
        *
        100
    )


    adversary_accuracy = (
        total_adversary_correct
        /
        total_samples
        *
        100
    )


    history_total.append(
        average_total
    )


    history_recon.append(
        average_recon
    )


    history_kl.append(
        average_kl
    )


    history_private_loss.append(
        average_private_loss
    )


    history_utility_loss.append(
        average_utility_loss
    )


    history_private_accuracy.append(
        private_accuracy
    )


    history_adversary_accuracy.append(
        adversary_accuracy
    )


    history_grl_lambda.append(
        current_grl_lambda
    )


    print()
    print("=" * 78)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} finished"
    )


    print(
        f"Average Total Loss: "
        f"{average_total:.6f}"
    )


    print(
        f"Average Reconstruction Loss: "
        f"{average_recon:.6f}"
    )


    print(
        f"Average KL Loss: "
        f"{average_kl:.6f}"
    )


    print(
        f"Average Private ID Loss: "
        f"{average_private_loss:.6f}"
    )


    print(
        f"Average Utility Adversarial Loss: "
        f"{average_utility_loss:.6f}"
    )


    print(
        f"Private Identity Accuracy: "
        f"{private_accuracy:.2f}%"
    )


    print(
        f"Utility Adversary Accuracy: "
        f"{adversary_accuracy:.2f}%"
    )


    print(
        f"Final GRL for epoch: "
        f"{current_grl_lambda:.4f}"
    )


    print("=" * 78)


    # ========================================================
    # 20. RECONSTRUCTION
    # ========================================================

    model.eval()


    with torch.no_grad():

        fixed_outputs = model(
            fixed_images,
            grl_lambda=current_grl_lambda
        )


        fixed_reconstructions = (
            fixed_outputs[
                "reconstruction"
            ]
        )


    save_reconstruction_grid(
        fixed_images.detach().cpu(),
        fixed_reconstructions.detach().cpu(),
        epoch
    )


    # ========================================================
    # 21. SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"adversarial_v2_epoch_{epoch:02d}.pth"
    )


    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "encoder_state_dict":
                model.encoder.state_dict(),

            "decoder_state_dict":
                model.decoder.state_dict(),

            "private_classifier_state_dict":
                model.private_classifier.state_dict(),

            "utility_adversary_state_dict":
                model.utility_identity_adversary.state_dict(),

            "main_optimizer_state_dict":
                main_optimizer.state_dict(),

            "adversary_optimizer_state_dict":
                adversary_optimizer.state_dict(),

            "latent_dim":
                LATENT_DIM,

            "private_dim":
                PRIVATE_DIM,

            "utility_dim":
                UTILITY_DIM,

            "num_identities":
                NUM_IDENTITIES,

            "beta_kl":
                BETA_KL,

            "lambda_private":
                LAMBDA_PRIVATE,

            "lambda_adversary":
                LAMBDA_ADVERSARY,

            "grl_max":
                GRL_MAX,

            "current_grl_lambda":
                current_grl_lambda,

            "average_total_loss":
                average_total,

            "average_reconstruction_loss":
                average_recon,

            "average_kl_loss":
                average_kl,

            "average_private_identity_loss":
                average_private_loss,

            "average_utility_identity_loss":
                average_utility_loss,

            "private_identity_accuracy":
                private_accuracy,

            "utility_adversary_accuracy":
                adversary_accuracy
        },

        checkpoint_path
    )


    print(
        f"Saved checkpoint to "
        f"{checkpoint_path}"
    )

    print()


# ============================================================
# 22. SAVE WARM-UP HISTORY
# ============================================================

warmup_csv_path = os.path.join(
    OUTPUT_DIR,
    "adversary_warmup_history.csv"
)


with open(
    warmup_csv_path,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )


    writer.writerow(
        [
            "epoch",
            "adversary_loss",
            "adversary_accuracy"
        ]
    )


    for i in range(
        ADVERSARY_WARMUP_EPOCHS
    ):

        writer.writerow(
            [
                i + 1,
                warmup_history_loss[i],
                warmup_history_accuracy[i]
            ]
        )


print(
    f"Saved warm-up history to "
    f"{warmup_csv_path}"
)


# ============================================================
# 23. SAVE ADVERSARIAL TRAINING HISTORY
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )


    writer.writerow(
        [
            "epoch",
            "total_loss",
            "reconstruction_loss",
            "kl_loss",
            "private_identity_loss",
            "utility_identity_loss",
            "private_identity_accuracy",
            "utility_adversary_accuracy",
            "grl_lambda"
        ]
    )


    for i in range(
        NUM_EPOCHS
    ):

        writer.writerow(
            [
                i + 1,
                history_total[i],
                history_recon[i],
                history_kl[i],
                history_private_loss[i],
                history_utility_loss[i],
                history_private_accuracy[i],
                history_adversary_accuracy[i],
                history_grl_lambda[i]
            ]
        )


print(
    f"Saved training history to "
    f"{csv_path}"
)


# ============================================================
# 24. WARM-UP ACCURACY GRAPH
# ============================================================

warmup_epochs = range(
    1,
    ADVERSARY_WARMUP_EPOCHS + 1
)


plt.figure(
    figsize=(9, 6)
)


plt.plot(
    warmup_epochs,
    warmup_history_accuracy,
    label="Utility Adversary Accuracy"
)


plt.axhline(
    y=100 / NUM_IDENTITIES,
    linestyle="--",
    label="Random Baseline"
)


plt.xlabel(
    "Warm-up Epoch"
)


plt.ylabel(
    "Identity Accuracy (%)"
)


plt.title(
    "Utility Adversary Warm-Up"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


warmup_plot_path = os.path.join(
    OUTPUT_DIR,
    "adversary_warmup_accuracy.png"
)


plt.savefig(
    warmup_plot_path,
    dpi=150
)


plt.close()


# ============================================================
# 25. LOSS COMPONENT GRAPH
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)


plt.figure(
    figsize=(9, 6)
)


plt.plot(
    epochs,
    history_recon,
    label="Reconstruction Loss"
)


plt.plot(
    epochs,
    history_private_loss,
    label="Private ID Loss"
)


plt.plot(
    epochs,
    history_utility_loss,
    label="Utility Adversarial Loss"
)


plt.plot(
    epochs,
    [
        BETA_KL * value
        for value
        in history_kl
    ],
    label="Weighted KL Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Phase 2D.4b Adversarial Loss Components"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "loss_components.png"
)


plt.savefig(
    loss_plot_path,
    dpi=150
)


plt.close()


# ============================================================
# 26. ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    epochs,
    history_private_accuracy,
    label="Private Identity Accuracy"
)


plt.plot(
    epochs,
    history_adversary_accuracy,
    label="Utility Adversary Accuracy"
)


plt.axhline(
    y=100 / NUM_IDENTITIES,
    linestyle="--",
    label="Random Baseline"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Identity Accuracy (%)"
)


plt.title(
    "Phase 2D.4b Identity Accuracy"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


accuracy_plot_path = os.path.join(
    OUTPUT_DIR,
    "identity_accuracy.png"
)


plt.savefig(
    accuracy_plot_path,
    dpi=150
)


plt.close()


# ============================================================
# 27. GRL GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    epochs,
    history_grl_lambda
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "GRL Lambda"
)


plt.title(
    "Gradient Reversal Strength"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


grl_plot_path = os.path.join(
    OUTPUT_DIR,
    "grl_lambda.png"
)


plt.savefig(
    grl_plot_path,
    dpi=150
)


plt.close()


print(
    f"Saved warm-up plot to "
    f"{warmup_plot_path}"
)

print(
    f"Saved loss plot to "
    f"{loss_plot_path}"
)

print(
    f"Saved accuracy plot to "
    f"{accuracy_plot_path}"
)

print(
    f"Saved GRL plot to "
    f"{grl_plot_path}"
)


# ============================================================
# 28. FINISH
# ============================================================

print()

print("=" * 78)

print(
    "Phase 2D.4b adversarial training completed."
)

print("=" * 78)

print()


print(
    "Warm-up adversary final accuracy:",
    f"{warmup_history_accuracy[-1]:.2f}%"
)


print()


print(
    "Final reconstruction loss:",
    f"{history_recon[-1]:.6f}"
)


print(
    "Final KL loss:",
    f"{history_kl[-1]:.6f}"
)


print(
    "Final private ID loss:",
    f"{history_private_loss[-1]:.6f}"
)


print(
    "Final utility adversarial loss:",
    f"{history_utility_loss[-1]:.6f}"
)


print(
    "Final private accuracy:",
    f"{history_private_accuracy[-1]:.2f}%"
)


print(
    "Final utility adversary accuracy:",
    f"{history_adversary_accuracy[-1]:.2f}%"
)


print(
    "Final GRL lambda:",
    f"{history_grl_lambda[-1]:.4f}"
)


print()


print(
    "Random-classification baseline:",
    f"{100 / NUM_IDENTITIES:.3f}%"
)


print()

print(
    "NEXT STEP:"
)

print(
    "Run the independent latent_identity_probe.py "
    "on adversarial_v2_epoch_05.pth."
)