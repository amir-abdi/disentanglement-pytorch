import os
import glob
import re
import csv
import random

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split
)

from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

from architectures.encoders.person_conv128x64 import (
    PersonGaussianConv128x64
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MARKET1501_TRAIN_DIR = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15/"
    "bounding_box_train"
)


# ============================================================
# CHECKPOINTS TO COMPARE
# ============================================================

# Baseline:
#
# Phase 2D.2b identity-aware model before utility metric
# learning.
#
# Compare against:
#
# Phase 2D.6 triplet + decorrelation checkpoints.

CHECKPOINTS = [
    {
        "name":
            "Baseline Private v2 Epoch 10",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_private_v2/"
                "checkpoints/"
                "private_v2_epoch_10.pth"
            )
    },

    {
        "name":
            "Triplet+Decorr Epoch 1",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_01.pth"
            )
    },

    {
        "name":
            "Triplet+Decorr Epoch 2",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_02.pth"
            )
    },

    {
        "name":
            "Triplet+Decorr Epoch 3",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_03.pth"
            )
    },

    {
        "name":
            "Triplet+Decorr Epoch 4",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_04.pth"
            )
    },

    {
        "name":
            "Triplet+Decorr Epoch 5",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_05.pth"
            )
    }
]


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
# 2. PROBE CONFIGURATION
# ============================================================

IMAGE_BATCH_SIZE = 64

PROBE_BATCH_SIZE = 64

NUM_WORKERS = 4

PROBE_EPOCHS = 15

PROBE_LR = 1e-3

TRAIN_FRACTION = 0.8

RANDOM_SEED = 42


# ============================================================
# 3. OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "latent_identity_probe_triplet_decorr_comparison"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "triplet_decorr_probe_results.csv"
)


PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "triplet_decorr_probe_comparison.png"
)


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

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


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


random_baseline = (
    100
    /
    NUM_IDENTITIES
)


print("=" * 80)

print(
    "Phase 2D.6 Triplet + Decorrelation "
    "Identity Leakage Analysis"
)

print("=" * 80)


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
    f"Random identity baseline: "
    f"{random_baseline:.3f}%"
)

print()


# ============================================================
# 6. DATASET
# ============================================================

class Market1501ProbeDataset(
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
                "No Market-1501 JPG images found in:\n"
                f"{image_dir}"
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


            if pid == -1:

                continue


            camid = int(
                match.group(2)
            )


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
            f"Loaded {len(self.samples)} images"
        )


        print(
            f"Loaded {len(self.pid2label)} identities"
        )


    def __len__(
        self
    ):

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
# 7. PREPROCESSING
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


dataset = Market1501ProbeDataset(
    MARKET1501_TRAIN_DIR,
    transform=transform
)


# ============================================================
# 8. FIXED TRAIN / TEST SPLIT
# ============================================================

train_size = int(
    len(dataset)
    *
    TRAIN_FRACTION
)


test_size = (
    len(dataset)
    -
    train_size
)


split_generator = (
    torch.Generator().manual_seed(
        RANDOM_SEED
    )
)


train_dataset, test_dataset = random_split(
    dataset,
    [
        train_size,
        test_size
    ],
    generator=split_generator
)


# ------------------------------------------------------------
# IMPORTANT:
#
# The split is generated ONCE.
#
# Every checkpoint therefore uses the exact same images for
# probe training and probe testing.
# ------------------------------------------------------------

train_image_loader = DataLoader(
    train_dataset,
    batch_size=IMAGE_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


test_image_loader = DataLoader(
    test_dataset,
    batch_size=IMAGE_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print(
    f"Probe training images: "
    f"{len(train_dataset)}"
)


print(
    f"Probe testing images: "
    f"{len(test_dataset)}"
)


print()


# ============================================================
# 9. BUILD ENCODER
# ============================================================

def build_encoder():

    encoder = PersonGaussianConv128x64(
        latent_dim=LATENT_DIM,
        num_channels=NUM_CHANNELS,
        image_size=None
    ).to(
        device
    )


    return encoder


# ============================================================
# 10. LOAD ENCODER CHECKPOINT
# ============================================================

def load_encoder(
    checkpoint_path
):

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}"
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )


    if (
        "encoder_state_dict"
        not in checkpoint
    ):

        raise KeyError(
            "Checkpoint does not contain "
            "'encoder_state_dict':\n"
            f"{checkpoint_path}"
        )


    encoder = build_encoder()


    encoder.load_state_dict(
        checkpoint[
            "encoder_state_dict"
        ]
    )


    encoder.eval()


    for parameter in encoder.parameters():

        parameter.requires_grad = False


    return encoder


# ============================================================
# 11. EXTRACT LATENT FEATURES
# ============================================================

def extract_features(
    encoder,
    data_loader
):

    private_features = []

    utility_features = []

    all_labels = []


    with torch.no_grad():

        for images, pids in data_loader:

            images = images.to(
                device,
                non_blocking=True
            )


            # ------------------------------------------------
            # Deterministic latent representation.
            # ------------------------------------------------

            mu, _ = encoder(
                images
            )


            # ------------------------------------------------
            # First 64 dimensions:
            #
            # mu_private
            # ------------------------------------------------

            mu_private = mu[
                :,
                :PRIVATE_DIM
            ]


            # ------------------------------------------------
            # Final 64 dimensions:
            #
            # mu_utility
            # ------------------------------------------------

            mu_utility = mu[
                :,
                PRIVATE_DIM:
            ]


            private_features.append(
                mu_private.cpu()
            )


            utility_features.append(
                mu_utility.cpu()
            )


            all_labels.append(
                pids.cpu()
            )


    private_features = torch.cat(
        private_features,
        dim=0
    )


    utility_features = torch.cat(
        utility_features,
        dim=0
    )


    labels = torch.cat(
        all_labels,
        dim=0
    )


    return (
        private_features,
        utility_features,
        labels
    )


# ============================================================
# 12. EVALUATE A TRAINED PROBE
# ============================================================

def evaluate_probe(
    probe,
    features,
    labels
):

    probe.eval()


    data = TensorDataset(
        features,
        labels
    )


    loader = DataLoader(
        data,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False
    )


    total_correct = 0

    total_samples = 0


    with torch.no_grad():

        for (
            feature_batch,
            pid_batch
        ) in loader:

            feature_batch = feature_batch.to(
                device
            )


            pid_batch = pid_batch.to(
                device
            )


            logits = probe(
                feature_batch
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            total_correct += (
                predictions
                ==
                pid_batch
            ).sum().item()


            total_samples += (
                pid_batch.size(0)
            )


    accuracy = (
        total_correct
        /
        total_samples
        *
        100
    )


    return accuracy


# ============================================================
# 13. TRAIN ONE LINEAR IDENTITY PROBE
# ============================================================

def train_probe(
    train_features,
    train_labels,
    test_features,
    test_labels,
    branch_name
):

    # --------------------------------------------------------
    # Reset the seed before every probe.
    #
    # Every checkpoint therefore gets equivalent probe
    # initialization and shuffled training order.
    # --------------------------------------------------------

    torch.manual_seed(
        RANDOM_SEED
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


    feature_dim = (
        train_features.shape[
            1
        ]
    )


    probe = nn.Linear(
        feature_dim,
        NUM_IDENTITIES
    ).to(
        device
    )


    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=PROBE_LR
    )


    criterion = nn.CrossEntropyLoss()


    train_data = TensorDataset(
        train_features,
        train_labels
    )


    shuffle_generator = (
        torch.Generator().manual_seed(
            RANDOM_SEED
        )
    )


    train_loader = DataLoader(
        train_data,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=True,
        generator=shuffle_generator
    )


    final_train_accuracy = 0.0

    final_test_accuracy = 0.0


    for epoch in range(
        1,
        PROBE_EPOCHS + 1
    ):

        probe.train()


        total_loss = 0.0

        total_correct = 0

        total_samples = 0


        for (
            feature_batch,
            pid_batch
        ) in train_loader:

            feature_batch = feature_batch.to(
                device
            )


            pid_batch = pid_batch.to(
                device
            )


            optimizer.zero_grad()


            logits = probe(
                feature_batch
            )


            loss = criterion(
                logits,
                pid_batch
            )


            loss.backward()


            optimizer.step()


            predictions = torch.argmax(
                logits,
                dim=1
            )


            total_correct += (
                predictions
                ==
                pid_batch
            ).sum().item()


            total_samples += (
                pid_batch.size(0)
            )


            total_loss += (
                loss.item()
            )


        average_loss = (
            total_loss
            /
            len(train_loader)
        )


        final_train_accuracy = (
            total_correct
            /
            total_samples
            *
            100
        )


        final_test_accuracy = evaluate_probe(
            probe,
            test_features,
            test_labels
        )


        # ----------------------------------------------------
        # Print selected epochs only.
        # ----------------------------------------------------

        if (
            epoch == 1
            or
            epoch == 5
            or
            epoch == 10
            or
            epoch == PROBE_EPOCHS
        ):

            print(
                f"  {branch_name:<7} "
                f"Epoch "
                f"[{epoch:02d}/{PROBE_EPOCHS}] | "
                f"Loss: "
                f"{average_loss:.4f} | "
                f"Train: "
                f"{final_train_accuracy:.2f}% | "
                f"Test: "
                f"{final_test_accuracy:.2f}%"
            )


    return (
        final_train_accuracy,
        final_test_accuracy
    )


# ============================================================
# 14. EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    checkpoint_name,
    checkpoint_path
):

    print()

    print("=" * 80)


    print(
        f"EVALUATING: "
        f"{checkpoint_name}"
    )


    print("=" * 80)


    print(
        checkpoint_path
    )


    print()


    # --------------------------------------------------------
    # Only the encoder is needed.
    #
    # This is an independent leakage test of what information
    # exists in mu_private and mu_utility.
    # --------------------------------------------------------

    encoder = load_encoder(
        checkpoint_path
    )


    print(
        "Extracting training latent features..."
    )


    (
        train_private,
        train_utility,
        train_labels
    ) = extract_features(
        encoder,
        train_image_loader
    )


    print(
        "Extracting testing latent features..."
    )


    (
        test_private,
        test_utility,
        test_labels
    ) = extract_features(
        encoder,
        test_image_loader
    )


    print()

    print(
        "Training private identity probe..."
    )


    (
        private_train_accuracy,
        private_test_accuracy
    ) = train_probe(
        train_private,
        train_labels,
        test_private,
        test_labels,
        branch_name="Private"
    )


    print()

    print(
        "Training utility identity probe..."
    )


    (
        utility_train_accuracy,
        utility_test_accuracy
    ) = train_probe(
        train_utility,
        train_labels,
        test_utility,
        test_labels,
        branch_name="Utility"
    )


    # --------------------------------------------------------
    # Clean up before loading next encoder.
    # --------------------------------------------------------

    del encoder


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    print()

    print(
        f"FINAL {checkpoint_name}"
    )


    print(
        f"Private test accuracy: "
        f"{private_test_accuracy:.2f}%"
    )


    print(
        f"Utility test accuracy: "
        f"{utility_test_accuracy:.2f}%"
    )


    return {
        "checkpoint":
            checkpoint_name,

        "path":
            checkpoint_path,

        "private_train_accuracy":
            private_train_accuracy,

        "private_test_accuracy":
            private_test_accuracy,

        "utility_train_accuracy":
            utility_train_accuracy,

        "utility_test_accuracy":
            utility_test_accuracy
    }


# ============================================================
# 15. RUN ALL CHECKPOINTS
# ============================================================

results = []


for checkpoint_info in CHECKPOINTS:

    result = evaluate_checkpoint(
        checkpoint_info[
            "name"
        ],
        checkpoint_info[
            "path"
        ]
    )


    results.append(
        result
    )


# ============================================================
# 16. FINAL COMPARISON TABLE
# ============================================================

print()

print("=" * 80)

print(
    "FINAL TRIPLET + DECORRELATION CHECKPOINT COMPARISON"
)

print("=" * 80)

print()


print(
    f"{'Checkpoint':<34}"
    f"{'Private':>12}"
    f"{'Utility':>12}"
)


print(
    "-" * 58
)


for result in results:

    print(
        f"{result['checkpoint']:<34}"
        f"{result['private_test_accuracy']:>11.2f}%"
        f"{result['utility_test_accuracy']:>11.2f}%"
    )


print(
    "-" * 58
)


print(
    f"{'Random baseline':<34}"
    f"{'':>12}"
    f"{random_baseline:>11.3f}%"
)


# ============================================================
# 17. PRIVACY SUMMARY
# ============================================================

baseline_utility = (
    results[
        0
    ][
        "utility_test_accuracy"
    ]
)


experimental_results = results[
    1:
]


best_privacy_result = min(
    experimental_results,
    key=lambda result:
        result[
            "utility_test_accuracy"
        ]
)


worst_privacy_result = max(
    experimental_results,
    key=lambda result:
        result[
            "utility_test_accuracy"
        ]
)


print()

print("=" * 80)

print(
    "TRIPLET + DECORRELATION PRIVACY SUMMARY"
)

print("=" * 80)

print()


print(
    f"Original baseline utility leakage: "
    f"{baseline_utility:.2f}%"
)


print()


print(
    f"Lowest-leakage experimental checkpoint: "
    f"{best_privacy_result['checkpoint']}"
)


print(
    f"Utility identity leakage: "
    f"{best_privacy_result['utility_test_accuracy']:.2f}%"
)


best_difference = (
    best_privacy_result[
        "utility_test_accuracy"
    ]
    -
    baseline_utility
)


print(
    f"Change from baseline: "
    f"{best_difference:+.2f} percentage points"
)


print()


print(
    f"Highest-leakage experimental checkpoint: "
    f"{worst_privacy_result['checkpoint']}"
)


print(
    f"Utility identity leakage: "
    f"{worst_privacy_result['utility_test_accuracy']:.2f}%"
)


worst_difference = (
    worst_privacy_result[
        "utility_test_accuracy"
    ]
    -
    baseline_utility
)


print(
    f"Change from baseline: "
    f"{worst_difference:+.2f} percentage points"
)


print()


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

if best_difference <= 2.0:

    print(
        "Interpretation: at least one Phase 2D.6 checkpoint "
        "kept utility identity leakage relatively close to "
        "the original baseline."
    )


elif best_difference < 15.0:

    print(
        "Interpretation: Phase 2D.6 increased utility identity "
        "leakage, but the increase is substantially more "
        "controlled than the earlier triplet-only experiment."
    )


else:

    print(
        "Interpretation: even the lowest-leakage Phase 2D.6 "
        "checkpoint contains substantially more absolute "
        "identity information than the original baseline."
    )


print()


print(
    "These privacy results should be interpreted together "
    "with cross-camera utility and private-utility dependence."
)


# ============================================================
# 18. SAVE RESULTS CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )


    writer.writerow(
        [
            "checkpoint",
            "checkpoint_path",
            "private_train_accuracy",
            "private_test_accuracy",
            "utility_train_accuracy",
            "utility_test_accuracy",
            "utility_leakage_change_from_baseline",
            "random_baseline"
        ]
    )


    for result in results:

        leakage_change = (
            result[
                "utility_test_accuracy"
            ]
            -
            baseline_utility
        )


        writer.writerow(
            [
                result[
                    "checkpoint"
                ],

                result[
                    "path"
                ],

                result[
                    "private_train_accuracy"
                ],

                result[
                    "private_test_accuracy"
                ],

                result[
                    "utility_train_accuracy"
                ],

                result[
                    "utility_test_accuracy"
                ],

                leakage_change,

                random_baseline
            ]
        )


print()

print(
    "Saved comparison CSV to:"
)


print(
    CSV_PATH
)


# ============================================================
# 19. COMPARISON GRAPH
# ============================================================

checkpoint_labels = [
    "Baseline",
    "T+D E1",
    "T+D E2",
    "T+D E3",
    "T+D E4",
    "T+D E5"
]


private_accuracies = [
    result[
        "private_test_accuracy"
    ]
    for result in results
]


utility_accuracies = [
    result[
        "utility_test_accuracy"
    ]
    for result in results
]


x_positions = list(
    range(
        len(
            checkpoint_labels
        )
    )
)


plt.figure(
    figsize=(11, 6)
)


plt.plot(
    x_positions,
    private_accuracies,
    marker="o",
    label="Private Probe Accuracy"
)


plt.plot(
    x_positions,
    utility_accuracies,
    marker="o",
    label="Utility Probe Accuracy"
)


plt.axhline(
    y=random_baseline,
    linestyle="--",
    label="Random Baseline"
)


plt.xticks(
    x_positions,
    checkpoint_labels
)


plt.xlabel(
    "Checkpoint"
)


plt.ylabel(
    "Identity Classification Accuracy (%)"
)


plt.title(
    "Absolute Identity Leakage Across "
    "Triplet + Decorrelation Checkpoints"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    PLOT_PATH,
    dpi=150
)


plt.close()


print(
    "Saved comparison plot to:"
)


print(
    PLOT_PATH
)


# ============================================================
# 20. FINISH
# ============================================================

print()

print("=" * 80)


print(
    "Phase 2D.6 identity leakage analysis completed."
)


print("=" * 80)