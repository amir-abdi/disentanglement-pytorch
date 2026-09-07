import os
import glob
import re
import csv
import random

import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader
)

from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image

import matplotlib.pyplot as plt


from architectures.encoders.person_conv128x64 import (
    PersonGaussianConv128x64
)

from architectures.decoders.person_conv128x64 import (
    PersonConv128x64
)

from common.ops import (
    kl_divergence_mu0_var1
)

from disentangled_person_vae import (
    DisentangledPersonVAE
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_SEED = 42


random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


MARKET1501_TRAIN_DIR = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15/"
    "bounding_box_train"
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Restart from our best pre-adversarial model.
# ------------------------------------------------------------

PRETRAINED_CHECKPOINT = (
    "outputs/"
    "market1501_disentangled_vae_private_v2/"
    "checkpoints/"
    "private_v2_epoch_10.pth"
)


IMAGE_HEIGHT = 128

IMAGE_WIDTH = 64

NUM_CHANNELS = 3


LATENT_DIM = 128

PRIVATE_DIM = 64

UTILITY_DIM = 64


assert (
    PRIVATE_DIM
    +
    UTILITY_DIM
    ==
    LATENT_DIM
)


NUM_IDENTITIES = 751


# ============================================================
# 2. LOSS CONFIGURATION
# ============================================================

BETA_KL = 0.001

LAMBDA_PRIVATE = 1.0


# ------------------------------------------------------------
# Cross-camera utility loss
#
# Start conservatively.
# ------------------------------------------------------------

LAMBDA_TRIPLET = 0.1


# Standard ReID-style margin.
TRIPLET_MARGIN = 0.3


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

# Each batch contains:
#
# anchor + positive + negative
#
# Therefore a batch size of 4 means the network processes
# 12 images per optimisation step.
#
# This is safer for the RTX 2050.
# ------------------------------------------------------------

BATCH_SIZE = 4


NUM_EPOCHS = 5


# Fine-tune gently because the existing representation
# already works well.
LEARNING_RATE = 5e-5


NUM_WORKERS = 4


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "market1501_disentangled_vae_triplet"
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
    "Phase 2D.5 - Cross-Camera Utility Triplet Learning"
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
    f"Private dimensions: "
    f"{PRIVATE_DIM}"
)

print(
    f"Utility dimensions: "
    f"{UTILITY_DIM}"
)

print()

print(
    f"BETA_KL: "
    f"{BETA_KL}"
)

print(
    f"LAMBDA_PRIVATE: "
    f"{LAMBDA_PRIVATE}"
)

print(
    f"LAMBDA_TRIPLET: "
    f"{LAMBDA_TRIPLET}"
)

print(
    f"TRIPLET_MARGIN: "
    f"{TRIPLET_MARGIN}"
)

print()


# ============================================================
# 5. CROSS-CAMERA TRIPLET DATASET
# ============================================================

class Market1501TripletDataset(
    Dataset
):

    def __init__(
        self,
        image_dir,
        transform=None
    ):

        self.transform = transform


        image_paths = sorted(
            glob.glob(
                os.path.join(
                    image_dir,
                    "*.jpg"
                )
            )
        )


        if len(
            image_paths
        ) == 0:

            raise RuntimeError(
                "No Market-1501 images found."
            )


        pattern = re.compile(
            r"([-\d]+)_c(\d)"
        )


        self.samples = []

        original_pids = []


        for image_path in image_paths:

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


        # ----------------------------------------------------
        # Relabel Market IDs:
        #
        # original PID -> 0 ... 750
        # ----------------------------------------------------

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


        self.pids = unique_pids


        # ----------------------------------------------------
        # Group samples by identity
        # ----------------------------------------------------

        self.pid_to_indices = {
            pid: []
            for pid in self.pids
        }


        for index, sample in enumerate(
            self.samples
        ):

            self.pid_to_indices[
                sample[
                    "original_pid"
                ]
            ].append(
                index
            )


        # ----------------------------------------------------
        # Verify that anchors have cross-camera positives.
        # ----------------------------------------------------

        self.valid_anchor_indices = []


        for index, sample in enumerate(
            self.samples
        ):

            pid = sample[
                "original_pid"
            ]


            camid = sample[
                "camid"
            ]


            positive_candidates = [
                candidate_index

                for candidate_index

                in self.pid_to_indices[
                    pid
                ]

                if self.samples[
                    candidate_index
                ][
                    "camid"
                ]
                != camid
            ]


            if len(
                positive_candidates
            ) > 0:

                self.valid_anchor_indices.append(
                    index
                )


        print(
            f"Loaded "
            f"{len(self.samples)} images"
        )


        print(
            f"Loaded "
            f"{len(self.pids)} identities"
        )


        print(
            f"Valid cross-camera anchors: "
            f"{len(self.valid_anchor_indices)}"
        )


    def __len__(
        self
    ):

        return len(
            self.valid_anchor_indices
        )


    def load_image(
        self,
        image_path
    ):

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        if self.transform is not None:

            image = self.transform(
                image
            )


        return image


    def __getitem__(
        self,
        index
    ):

        # ====================================================
        # ANCHOR
        # ====================================================

        anchor_index = (
            self.valid_anchor_indices[
                index
            ]
        )


        anchor_sample = (
            self.samples[
                anchor_index
            ]
        )


        anchor_pid = anchor_sample[
            "original_pid"
        ]


        anchor_camid = anchor_sample[
            "camid"
        ]


        # ====================================================
        # POSITIVE
        #
        # Same identity
        # Different camera
        # ====================================================

        positive_candidates = [
            candidate_index

            for candidate_index

            in self.pid_to_indices[
                anchor_pid
            ]

            if self.samples[
                candidate_index
            ][
                "camid"
            ]
            != anchor_camid
        ]


        positive_index = random.choice(
            positive_candidates
        )


        positive_sample = (
            self.samples[
                positive_index
            ]
        )


        # ====================================================
        # NEGATIVE
        #
        # Different identity
        # ====================================================

        negative_pid = random.choice(
            self.pids
        )


        while (
            negative_pid
            ==
            anchor_pid
        ):

            negative_pid = random.choice(
                self.pids
            )


        negative_index = random.choice(
            self.pid_to_indices[
                negative_pid
            ]
        )


        negative_sample = (
            self.samples[
                negative_index
            ]
        )


        # ====================================================
        # LOAD IMAGES
        # ====================================================

        anchor_image = self.load_image(
            anchor_sample[
                "path"
            ]
        )


        positive_image = self.load_image(
            positive_sample[
                "path"
            ]
        )


        negative_image = self.load_image(
            negative_sample[
                "path"
            ]
        )


        # ====================================================
        # CLASSIFICATION LABELS
        # ====================================================

        anchor_label = self.pid2label[
            anchor_pid
        ]


        positive_label = anchor_label


        negative_label = self.pid2label[
            negative_pid
        ]


        return (
            anchor_image,
            positive_image,
            negative_image,
            anchor_label,
            positive_label,
            negative_label
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


dataset = Market1501TripletDataset(
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
# 7. FIXED RECONSTRUCTION EXAMPLES
# ============================================================

fixed_triplets = [
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
        triplet[0]
        for triplet
        in fixed_triplets
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
    num_identities=NUM_IDENTITIES
).to(
    device
)


# ============================================================
# 9. LOAD BEST BASELINE CHECKPOINT
# ============================================================

print(
    "Loading Phase 2D.2b baseline..."
)


checkpoint = torch.load(
    PRETRAINED_CHECKPOINT,
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
    "Loaded encoder."
)

print(
    "Loaded decoder."
)

print(
    "Loaded private classifier."
)

print()

print(
    PRETRAINED_CHECKPOINT
)

print()


# ============================================================
# 10. OPTIMIZER
# ============================================================

# IMPORTANT:
#
# Utility adversary is NOT included.
#
# We are not performing adversarial training in Phase 2D.5.
# ------------------------------------------------------------

optimizer = torch.optim.Adam(
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
    lr=LEARNING_RATE
)


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

def compute_losses(
    reconstructed,
    original,
    mu,
    logvar,
    private_logits,
    labels,
    anchor_utility,
    positive_utility,
    negative_utility
):

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    recon_loss = (
        F.binary_cross_entropy(
            reconstructed,
            original,
            reduction="mean"
        )
    )


    # --------------------------------------------------------
    # KL
    # --------------------------------------------------------

    kl_loss = (
        kl_divergence_mu0_var1(
            mu,
            logvar
        )
    )


    # --------------------------------------------------------
    # Private identity classification
    # --------------------------------------------------------

    private_loss = (
        F.cross_entropy(
            private_logits,
            labels
        )
    )


    # --------------------------------------------------------
    # NORMALISE UTILITY REPRESENTATIONS
    #
    # This prevents raw feature magnitude from dominating
    # the triplet objective.
    # --------------------------------------------------------

    anchor_utility = F.normalize(
        anchor_utility,
        p=2,
        dim=1
    )


    positive_utility = F.normalize(
        positive_utility,
        p=2,
        dim=1
    )


    negative_utility = F.normalize(
        negative_utility,
        p=2,
        dim=1
    )


    # --------------------------------------------------------
    # Cross-camera triplet loss
    # --------------------------------------------------------

    triplet_loss = (
        F.triplet_margin_loss(
            anchor_utility,
            positive_utility,
            negative_utility,
            margin=TRIPLET_MARGIN,
            p=2,
            reduction="mean"
        )
    )


    # --------------------------------------------------------
    # Total Phase 2D.5 objective
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
        private_loss
        +
        LAMBDA_TRIPLET
        *
        triplet_loss
    )


    return (
        total_loss,
        recon_loss,
        kl_loss,
        private_loss,
        triplet_loss
    )


# ============================================================
# 12. HISTORY
# ============================================================

history_total = []

history_recon = []

history_kl = []

history_private = []

history_triplet = []

history_private_accuracy = []

history_triplet_ordering = []


# ============================================================
# 13. TRAINING
# ============================================================

print("=" * 78)

print(
    "Starting Phase 2D.5 training..."
)

print("=" * 78)

print()


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    model.train()


    total_loss_value = 0.0

    total_recon = 0.0

    total_kl = 0.0

    total_private = 0.0

    total_triplet = 0.0


    private_correct = 0

    private_samples = 0


    correctly_ordered_triplets = 0

    total_triplets = 0


    for batch_idx, batch in enumerate(
        data_loader
    ):

        (
            anchor_images,
            positive_images,
            negative_images,
            anchor_labels,
            positive_labels,
            negative_labels
        ) = batch


        anchor_images = anchor_images.to(
            device,
            non_blocking=True
        )


        positive_images = positive_images.to(
            device,
            non_blocking=True
        )


        negative_images = negative_images.to(
            device,
            non_blocking=True
        )


        anchor_labels = anchor_labels.to(
            device
        )


        positive_labels = positive_labels.to(
            device
        )


        negative_labels = negative_labels.to(
            device
        )


        # ====================================================
        # COMBINE ALL 3 IMAGE GROUPS
        # ====================================================

        all_images = torch.cat(
            [
                anchor_images,
                positive_images,
                negative_images
            ],
            dim=0
        )


        all_labels = torch.cat(
            [
                anchor_labels,
                positive_labels,
                negative_labels
            ],
            dim=0
        )


        optimizer.zero_grad()


        # ====================================================
        # ENCODE
        # ====================================================

        latent_outputs = model.encode(
            all_images
        )


        mu = latent_outputs[
            "mu"
        ]


        logvar = latent_outputs[
            "logvar"
        ]


        mu_private = latent_outputs[
            "mu_private"
        ]


        mu_utility = latent_outputs[
            "mu_utility"
        ]


        z_private = latent_outputs[
            "z_private"
        ]


        z_utility = latent_outputs[
            "z_utility"
        ]


        # ====================================================
        # RECONSTRUCTION
        # ====================================================

        reconstructed = model.decode(
            z_private,
            z_utility
        )


        # ====================================================
        # PRIVATE IDENTITY CLASSIFICATION
        # ====================================================

        private_logits = (
            model.classify_private(
                mu_private
            )
        )


        # ====================================================
        # SPLIT UTILITY REPRESENTATIONS
        #
        # Order is:
        #
        # anchor
        # positive
        # negative
        # ====================================================

        current_batch_size = (
            anchor_images.size(0)
        )


        anchor_utility = mu_utility[
            0:
            current_batch_size
        ]


        positive_utility = mu_utility[
            current_batch_size:
            2 * current_batch_size
        ]


        negative_utility = mu_utility[
            2 * current_batch_size:
            3 * current_batch_size
        ]


        # ====================================================
        # LOSSES
        # ====================================================

        (
            loss,
            recon_loss,
            kl_loss,
            private_loss,
            triplet_loss
        ) = compute_losses(
            reconstructed,
            all_images,
            mu,
            logvar,
            private_logits,
            all_labels,
            anchor_utility,
            positive_utility,
            negative_utility
        )


        loss.backward()


        optimizer.step()


        # ====================================================
        # PRIVATE ACCURACY
        # ====================================================

        private_predictions = (
            torch.argmax(
                private_logits,
                dim=1
            )
        )


        private_correct += (
            private_predictions
            ==
            all_labels
        ).sum().item()


        private_samples += (
            all_labels.size(0)
        )


        # ====================================================
        # TRIPLET ORDERING ACCURACY
        #
        # Is the positive actually closer than the negative?
        # ====================================================

        with torch.no_grad():

            normalized_anchor = (
                F.normalize(
                    anchor_utility,
                    p=2,
                    dim=1
                )
            )


            normalized_positive = (
                F.normalize(
                    positive_utility,
                    p=2,
                    dim=1
                )
            )


            normalized_negative = (
                F.normalize(
                    negative_utility,
                    p=2,
                    dim=1
                )
            )


            positive_distance = torch.norm(
                normalized_anchor
                -
                normalized_positive,
                p=2,
                dim=1
            )


            negative_distance = torch.norm(
                normalized_anchor
                -
                normalized_negative,
                p=2,
                dim=1
            )


            correctly_ordered_triplets += (
                positive_distance
                <
                negative_distance
            ).sum().item()


            total_triplets += (
                current_batch_size
            )


        # ====================================================
        # STATISTICS
        # ====================================================

        total_loss_value += (
            loss.item()
        )


        total_recon += (
            recon_loss.item()
        )


        total_kl += (
            kl_loss.item()
        )


        total_private += (
            private_loss.item()
        )


        total_triplet += (
            triplet_loss.item()
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
                "PHASE 2D.5 MODEL CHECK"
            )

            print(
                "=" * 70
            )


            print(
                "Anchor images:",
                tuple(
                    anchor_images.shape
                )
            )


            print(
                "All images:",
                tuple(
                    all_images.shape
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
                "Anchor utility:",
                tuple(
                    anchor_utility.shape
                )
            )


            print(
                "Positive utility:",
                tuple(
                    positive_utility.shape
                )
            )


            print(
                "Negative utility:",
                tuple(
                    negative_utility.shape
                )
            )


            print(
                "=" * 70
            )

            print()


        # ====================================================
        # PROGRESS
        # ====================================================

        if batch_idx % 100 == 0:

            batch_private_accuracy = (
                (
                    private_predictions
                    ==
                    all_labels
                )
                .float()
                .mean()
                .item()
                *
                100
            )


            batch_ordering = (
                (
                    positive_distance
                    <
                    negative_distance
                )
                .float()
                .mean()
                .item()
                *
                100
            )


            print(
                f"Epoch "
                f"[{epoch:02d}/{NUM_EPOCHS}] "
                f"Batch "
                f"[{batch_idx:04d}/"
                f"{len(data_loader)}] "
                f"Total: "
                f"{loss.item():.4f} "
                f"Recon: "
                f"{recon_loss.item():.4f} "
                f"KL: "
                f"{kl_loss.item():.2f} "
                f"Private-ID: "
                f"{private_loss.item():.4f} "
                f"Triplet: "
                f"{triplet_loss.item():.4f} "
                f"Private-Acc: "
                f"{batch_private_accuracy:.2f}% "
                f"Triplet-Order: "
                f"{batch_ordering:.2f}%"
            )


    # ========================================================
    # 14. EPOCH RESULTS
    # ========================================================

    number_batches = len(
        data_loader
    )


    average_total = (
        total_loss_value
        /
        number_batches
    )


    average_recon = (
        total_recon
        /
        number_batches
    )


    average_kl = (
        total_kl
        /
        number_batches
    )


    average_private = (
        total_private
        /
        number_batches
    )


    average_triplet = (
        total_triplet
        /
        number_batches
    )


    private_accuracy = (
        private_correct
        /
        private_samples
        *
        100
    )


    triplet_ordering_accuracy = (
        correctly_ordered_triplets
        /
        total_triplets
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


    history_private.append(
        average_private
    )


    history_triplet.append(
        average_triplet
    )


    history_private_accuracy.append(
        private_accuracy
    )


    history_triplet_ordering.append(
        triplet_ordering_accuracy
    )


    print()

    print("=" * 78)


    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
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
        f"{average_private:.6f}"
    )


    print(
        f"Average Utility Triplet Loss: "
        f"{average_triplet:.6f}"
    )


    print(
        f"Private Accuracy: "
        f"{private_accuracy:.2f}%"
    )


    print(
        f"Triplet Ordering Accuracy: "
        f"{triplet_ordering_accuracy:.2f}%"
    )


    print("=" * 78)

    print()


    # ========================================================
    # 15. RECONSTRUCTION
    # ========================================================

    model.eval()


    with torch.no_grad():

        fixed_latents = model.encode(
            fixed_images
        )


        fixed_reconstructions = (
            model.decode(
                fixed_latents[
                    "z_private"
                ],
                fixed_latents[
                    "z_utility"
                ]
            )
        )


    comparison = torch.cat(
        [
            fixed_images.cpu(),
            fixed_reconstructions.cpu()
        ],
        dim=0
    )


    reconstruction_path = os.path.join(
        RECON_DIR,
        f"epoch_{epoch:02d}.png"
    )


    save_image(
        comparison,
        reconstruction_path,
        nrow=fixed_images.size(0)
    )


    # ========================================================
    # 16. SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"triplet_epoch_{epoch:02d}.pth"
    )


    torch.save(
        {
            "epoch":
                epoch,

            "encoder_state_dict":
                model.encoder.state_dict(),

            "decoder_state_dict":
                model.decoder.state_dict(),

            "private_classifier_state_dict":
                model.private_classifier.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "latent_dim":
                LATENT_DIM,

            "private_dim":
                PRIVATE_DIM,

            "utility_dim":
                UTILITY_DIM,

            "beta_kl":
                BETA_KL,

            "lambda_private":
                LAMBDA_PRIVATE,

            "lambda_triplet":
                LAMBDA_TRIPLET,

            "triplet_margin":
                TRIPLET_MARGIN,

            "average_total_loss":
                average_total,

            "average_reconstruction_loss":
                average_recon,

            "average_kl_loss":
                average_kl,

            "average_private_loss":
                average_private,

            "average_triplet_loss":
                average_triplet,

            "private_accuracy":
                private_accuracy,

            "triplet_ordering_accuracy":
                triplet_ordering_accuracy
        },

        checkpoint_path
    )


    print(
        f"Saved checkpoint to:"
    )


    print(
        checkpoint_path
    )

    print()


# ============================================================
# 17. SAVE TRAINING HISTORY
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
            "triplet_loss",
            "private_accuracy",
            "triplet_ordering_accuracy"
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
                history_private[i],
                history_triplet[i],
                history_private_accuracy[i],
                history_triplet_ordering[i]
            ]
        )


# ============================================================
# 18. LOSS GRAPH
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
    history_private,
    label="Private ID Loss"
)


plt.plot(
    epochs,
    history_triplet,
    label="Utility Triplet Loss"
)


plt.plot(
    epochs,
    [
        BETA_KL * value
        for value
        in history_kl
    ],
    label="Weighted KL"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Phase 2D.5 Loss Components"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_components.png"
    ),
    dpi=150
)


plt.close()


# ============================================================
# 19. TRIPLET ORDERING GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    epochs,
    history_triplet_ordering
)


plt.axhline(
    y=50,
    linestyle="--",
    label="Random ordering"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Correct Triplet Ordering (%)"
)


plt.title(
    "Cross-Camera Utility Triplet Ordering"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "triplet_ordering_accuracy.png"
    ),
    dpi=150
)


plt.close()


# ============================================================
# 20. FINISH
# ============================================================

print()

print("=" * 78)

print(
    "Phase 2D.5 training completed."
)

print("=" * 78)

print()


print(
    f"Final reconstruction loss: "
    f"{history_recon[-1]:.6f}"
)


print(
    f"Final private accuracy: "
    f"{history_private_accuracy[-1]:.2f}%"
)


print(
    f"Final triplet loss: "
    f"{history_triplet[-1]:.6f}"
)


print(
    f"Final triplet ordering accuracy: "
    f"{history_triplet_ordering[-1]:.2f}%"
)


print()

print(
    "Final checkpoint:"
)


print(
    os.path.join(
        CHECKPOINT_DIR,
        f"triplet_epoch_{NUM_EPOCHS:02d}.pth"
    )
)


print()

print(
    "NEXT:"
)

print(
    "1. Re-run utility cross-camera analysis."
)

print(
    "2. Re-run independent identity leakage probe."
)