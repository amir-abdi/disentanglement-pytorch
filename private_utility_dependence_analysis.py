import os
import glob
import re
import csv
import random

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader
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


BATCH_SIZE = 64

NUM_WORKERS = 4

RANDOM_SEED = 42


# ============================================================
# 2. CHECKPOINTS
# ============================================================

# ------------------------------------------------------------
# Known independent utility PID probe accuracies
#
# These come from latent_identity_probe.py.
#
# They are NOT training accuracies.
# ------------------------------------------------------------

CHECKPOINTS = [
    {
        "name":
            "Baseline",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_private_v2/"
                "checkpoints/"
                "private_v2_epoch_10.pth"
            ),

        "utility_probe_accuracy":
            5.29
    },

    {
        "name":
            "T+D E1",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_01.pth"
            ),

        "utility_probe_accuracy":
            16.92
    },

    {
        "name":
            "T+D E2",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_02.pth"
            ),

        "utility_probe_accuracy":
            17.54
    },

    {
        "name":
            "T+D E3",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_03.pth"
            ),

        "utility_probe_accuracy":
            18.59
    },

    {
        "name":
            "T+D E4",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_04.pth"
            ),

        "utility_probe_accuracy":
            17.70
    },

    {
        "name":
            "T+D E5",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_05.pth"
            ),

        "utility_probe_accuracy":
            16.15
    }
]


# ============================================================
# 3. REFERENCE VALUES FROM TRIPLET-ONLY EXPERIMENT
# ============================================================

# These are useful for interpretation at the end.
#
# Triplet-only Epoch 5 previously produced:
#
# Utility PID leakage = 71.17%
# Linear CKA          = 0.420055
# Mean |r|            = 0.090939

TRIPLET_ONLY_E5_PID_LEAKAGE = 71.17

TRIPLET_ONLY_E5_CKA = 0.420055

TRIPLET_ONLY_E5_MEAN_ABS_CORR = 0.090939


# ============================================================
# 4. OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "private_utility_dependence_triplet_decorr"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "triplet_decorr_private_utility_dependence.csv"
)


CKA_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "triplet_decorr_linear_cka.png"
)


CORRELATION_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "triplet_decorr_cross_correlation.png"
)


LEAKAGE_CKA_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "cka_vs_identity_leakage.png"
)


LEAKAGE_CORRELATION_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "correlation_vs_identity_leakage.png"
)


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)


np.random.seed(
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
# 6. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)

print(
    "Phase 2D.6 Private-Utility Dependence Analysis"
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


# ============================================================
# 7. DATASET
# ============================================================

class Market1501DependenceDataset(
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

                    "pid":
                        pid,

                    "camid":
                        camid
                }
            )


        unique_pids = set(
            sample[
                "pid"
            ]
            for sample in self.samples
        )


        print(
            f"Loaded "
            f"{len(self.samples)} images"
        )


        print(
            f"Loaded "
            f"{len(unique_pids)} identities"
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


        return image


# ============================================================
# 8. PREPROCESSING
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


dataset = Market1501DependenceDataset(
    MARKET1501_TRAIN_DIR,
    transform=transform
)


data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
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
# 10. LOAD ENCODER
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
# 11. EXTRACT LATENT REPRESENTATIONS
# ============================================================

def extract_latents(
    encoder
):

    private_latents = []

    utility_latents = []


    with torch.no_grad():

        for images in data_loader:

            images = images.to(
                device,
                non_blocking=True
            )


            # ------------------------------------------------
            # Deterministic VAE latent mean.
            # ------------------------------------------------

            mu, _ = encoder(
                images
            )


            mu_private = mu[
                :,
                :PRIVATE_DIM
            ]


            mu_utility = mu[
                :,
                PRIVATE_DIM:
            ]


            private_latents.append(
                mu_private.cpu()
            )


            utility_latents.append(
                mu_utility.cpu()
            )


    private_latents = torch.cat(
        private_latents,
        dim=0
    ).float()


    utility_latents = torch.cat(
        utility_latents,
        dim=0
    ).float()


    return (
        private_latents,
        utility_latents
    )


# ============================================================
# 12. CENTER FEATURES
# ============================================================

def center_features(
    features
):

    centered = (
        features
        -
        features.mean(
            dim=0,
            keepdim=True
        )
    )


    return centered


# ============================================================
# 13. CROSS-COVARIANCE MATRIX
# ============================================================

def cross_covariance_matrix(
    private_features,
    utility_features
):

    private_centered = center_features(
        private_features
    )


    utility_centered = center_features(
        utility_features
    )


    number_samples = (
        private_centered.shape[
            0
        ]
    )


    cross_covariance = (
        private_centered.T
        @
        utility_centered
    ) / (
        number_samples
        -
        1
    )


    return cross_covariance


# ============================================================
# 14. CROSS-CORRELATION MATRIX
# ============================================================

def cross_correlation_matrix(
    private_features,
    utility_features,
    epsilon=1e-8
):

    private_centered = center_features(
        private_features
    )


    utility_centered = center_features(
        utility_features
    )


    private_std = private_centered.std(
        dim=0,
        unbiased=True
    )


    utility_std = utility_centered.std(
        dim=0,
        unbiased=True
    )


    private_std = torch.clamp(
        private_std,
        min=epsilon
    )


    utility_std = torch.clamp(
        utility_std,
        min=epsilon
    )


    private_standardized = (
        private_centered
        /
        private_std
    )


    utility_standardized = (
        utility_centered
        /
        utility_std
    )


    number_samples = (
        private_features.shape[
            0
        ]
    )


    correlation = (
        private_standardized.T
        @
        utility_standardized
    ) / (
        number_samples
        -
        1
    )


    return correlation


# ============================================================
# 15. LINEAR CKA
# ============================================================

def linear_cka(
    private_features,
    utility_features,
    epsilon=1e-12
):

    private_centered = center_features(
        private_features
    )


    utility_centered = center_features(
        utility_features
    )


    # --------------------------------------------------------
    # Cross representation product.
    # --------------------------------------------------------

    private_utility = (
        private_centered.T
        @
        utility_centered
    )


    # --------------------------------------------------------
    # Within-representation products.
    # --------------------------------------------------------

    private_private = (
        private_centered.T
        @
        private_centered
    )


    utility_utility = (
        utility_centered.T
        @
        utility_centered
    )


    # --------------------------------------------------------
    # Linear CKA numerator.
    # --------------------------------------------------------

    numerator = torch.sum(
        private_utility
        **
        2
    )


    # --------------------------------------------------------
    # Linear CKA denominator.
    # --------------------------------------------------------

    private_norm = torch.sqrt(
        torch.sum(
            private_private
            **
            2
        )
    )


    utility_norm = torch.sqrt(
        torch.sum(
            utility_utility
            **
            2
        )
    )


    denominator = (
        private_norm
        *
        utility_norm
    )


    cka = (
        numerator
        /
        torch.clamp(
            denominator,
            min=epsilon
        )
    )


    return cka.item()


# ============================================================
# 16. COMPUTE DEPENDENCE METRICS
# ============================================================

def compute_dependence_metrics(
    private_features,
    utility_features
):

    covariance = cross_covariance_matrix(
        private_features,
        utility_features
    )


    correlation = cross_correlation_matrix(
        private_features,
        utility_features
    )


    # --------------------------------------------------------
    # Cross-covariance Frobenius norm
    #
    # Scale-sensitive diagnostic.
    # --------------------------------------------------------

    covariance_frobenius = (
        torch.linalg.norm(
            covariance,
            ord="fro"
        )
        .item()
    )


    # --------------------------------------------------------
    # Absolute cross-correlations.
    # --------------------------------------------------------

    absolute_correlation = torch.abs(
        correlation
    )


    mean_absolute_correlation = (
        absolute_correlation
        .mean()
        .item()
    )


    rms_correlation = (
        torch.sqrt(
            torch.mean(
                correlation
                **
                2
            )
        )
        .item()
    )


    maximum_absolute_correlation = (
        absolute_correlation
        .max()
        .item()
    )


    # --------------------------------------------------------
    # Linear representation similarity.
    # --------------------------------------------------------

    cka = linear_cka(
        private_features,
        utility_features
    )


    return {
        "cross_covariance_frobenius":
            covariance_frobenius,

        "mean_absolute_cross_correlation":
            mean_absolute_correlation,

        "rms_cross_correlation":
            rms_correlation,

        "max_absolute_cross_correlation":
            maximum_absolute_correlation,

        "linear_cka":
            cka
    }


# ============================================================
# 17. EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    checkpoint_info
):

    checkpoint_name = (
        checkpoint_info[
            "name"
        ]
    )


    checkpoint_path = (
        checkpoint_info[
            "path"
        ]
    )


    utility_probe_accuracy = (
        checkpoint_info[
            "utility_probe_accuracy"
        ]
    )


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


    encoder = load_encoder(
        checkpoint_path
    )


    print(
        "Extracting latent representations..."
    )


    (
        private_features,
        utility_features
    ) = extract_latents(
        encoder
    )


    print(
        "Private latent shape:",
        tuple(
            private_features.shape
        )
    )


    print(
        "Utility latent shape:",
        tuple(
            utility_features.shape
        )
    )


    print(
        "Computing full-dataset dependence metrics..."
    )


    metrics = compute_dependence_metrics(
        private_features,
        utility_features
    )


    print()


    print(
        f"Cross-covariance Frobenius norm: "
        f"{metrics['cross_covariance_frobenius']:.6f}"
    )


    print(
        f"Mean absolute cross-correlation: "
        f"{metrics['mean_absolute_cross_correlation']:.6f}"
    )


    print(
        f"RMS cross-correlation: "
        f"{metrics['rms_cross_correlation']:.6f}"
    )


    print(
        f"Maximum absolute cross-correlation: "
        f"{metrics['max_absolute_cross_correlation']:.6f}"
    )


    print(
        f"Linear CKA: "
        f"{metrics['linear_cka']:.6f}"
    )


    print(
        f"Known utility PID leakage: "
        f"{utility_probe_accuracy:.2f}%"
    )


    print()


    result = {
        "checkpoint":
            checkpoint_name,

        "checkpoint_path":
            checkpoint_path,

        "utility_probe_accuracy":
            utility_probe_accuracy,

        **metrics
    }


    del encoder

    del private_features

    del utility_features


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# 18. RUN ALL CHECKPOINTS
# ============================================================

results = []


for checkpoint_info in CHECKPOINTS:

    result = evaluate_checkpoint(
        checkpoint_info
    )


    results.append(
        result
    )


# ============================================================
# 19. FINAL TABLE
# ============================================================

print()

print("=" * 112)


print(
    "FINAL PHASE 2D.6 PRIVATE-UTILITY DEPENDENCE COMPARISON"
)


print("=" * 112)

print()


print(
    f"{'Checkpoint':<14}"
    f"{'PID Leak':>12}"
    f"{'Cov Fro':>14}"
    f"{'Mean |r|':>14}"
    f"{'RMS r':>14}"
    f"{'Max |r|':>14}"
    f"{'CKA':>14}"
)


print(
    "-" * 96
)


for result in results:

    print(
        f"{result['checkpoint']:<14}"
        f"{result['utility_probe_accuracy']:>11.2f}%"
        f"{result['cross_covariance_frobenius']:>14.4f}"
        f"{result['mean_absolute_cross_correlation']:>14.4f}"
        f"{result['rms_cross_correlation']:>14.4f}"
        f"{result['max_absolute_cross_correlation']:>14.4f}"
        f"{result['linear_cka']:>14.4f}"
    )


# ============================================================
# 20. BASELINE / FINAL COMPARISON
# ============================================================

baseline_result = results[
    0
]


final_result = results[
    -1
]


print()

print("=" * 80)

print(
    "BASELINE VS PHASE 2D.6 EPOCH 5"
)

print("=" * 80)

print()


print(
    f"Utility PID leakage:"
)


print(
    f"  Baseline: "
    f"{baseline_result['utility_probe_accuracy']:.2f}%"
)


print(
    f"  Phase 2D.6 E5: "
    f"{final_result['utility_probe_accuracy']:.2f}%"
)


print()


print(
    "Linear CKA:"
)


print(
    f"  Baseline: "
    f"{baseline_result['linear_cka']:.6f}"
)


print(
    f"  Phase 2D.6 E5: "
    f"{final_result['linear_cka']:.6f}"
)


print()


print(
    "Mean absolute cross-correlation:"
)


print(
    f"  Baseline: "
    f"{baseline_result['mean_absolute_cross_correlation']:.6f}"
)


print(
    f"  Phase 2D.6 E5: "
    f"{final_result['mean_absolute_cross_correlation']:.6f}"
)


# ============================================================
# 21. COMPARE AGAINST TRIPLET-ONLY EPOCH 5
# ============================================================

print()

print("=" * 80)

print(
    "PHASE 2D.6 EPOCH 5 VS TRIPLET-ONLY EPOCH 5"
)

print("=" * 80)

print()


print(
    "Utility PID leakage:"
)


print(
    f"  Triplet-only E5: "
    f"{TRIPLET_ONLY_E5_PID_LEAKAGE:.2f}%"
)


print(
    f"  Triplet+Decorr E5: "
    f"{final_result['utility_probe_accuracy']:.2f}%"
)


pid_reduction = (
    TRIPLET_ONLY_E5_PID_LEAKAGE
    -
    final_result[
        "utility_probe_accuracy"
    ]
)


print(
    f"  Leakage reduction: "
    f"{pid_reduction:.2f} percentage points"
)


print()


print(
    "Linear CKA:"
)


print(
    f"  Triplet-only E5: "
    f"{TRIPLET_ONLY_E5_CKA:.6f}"
)


print(
    f"  Triplet+Decorr E5: "
    f"{final_result['linear_cka']:.6f}"
)


cka_change = (
    final_result[
        "linear_cka"
    ]
    -
    TRIPLET_ONLY_E5_CKA
)


print(
    f"  Change: "
    f"{cka_change:+.6f}"
)


print()


print(
    "Mean absolute cross-correlation:"
)


print(
    f"  Triplet-only E5: "
    f"{TRIPLET_ONLY_E5_MEAN_ABS_CORR:.6f}"
)


print(
    f"  Triplet+Decorr E5: "
    f"{final_result['mean_absolute_cross_correlation']:.6f}"
)


correlation_change = (
    final_result[
        "mean_absolute_cross_correlation"
    ]
    -
    TRIPLET_ONLY_E5_MEAN_ABS_CORR
)


print(
    f"  Change: "
    f"{correlation_change:+.6f}"
)


# ============================================================
# 22. SAVE RESULTS CSV
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
            "utility_probe_accuracy",
            "cross_covariance_frobenius",
            "mean_absolute_cross_correlation",
            "rms_cross_correlation",
            "max_absolute_cross_correlation",
            "linear_cka"
        ]
    )


    for result in results:

        writer.writerow(
            [
                result[
                    "checkpoint"
                ],

                result[
                    "checkpoint_path"
                ],

                result[
                    "utility_probe_accuracy"
                ],

                result[
                    "cross_covariance_frobenius"
                ],

                result[
                    "mean_absolute_cross_correlation"
                ],

                result[
                    "rms_cross_correlation"
                ],

                result[
                    "max_absolute_cross_correlation"
                ],

                result[
                    "linear_cka"
                ]
            ]
        )


print()

print(
    "Saved results CSV to:"
)


print(
    CSV_PATH
)


# ============================================================
# 23. PREPARE PLOT VALUES
# ============================================================

checkpoint_names = [
    result[
        "checkpoint"
    ]
    for result in results
]


x_positions = list(
    range(
        len(
            checkpoint_names
        )
    )
)


cka_values = [
    result[
        "linear_cka"
    ]
    for result in results
]


mean_correlations = [
    result[
        "mean_absolute_cross_correlation"
    ]
    for result in results
]


rms_correlations = [
    result[
        "rms_cross_correlation"
    ]
    for result in results
]


utility_probe_accuracies = [
    result[
        "utility_probe_accuracy"
    ]
    for result in results
]


# ============================================================
# 24. LINEAR CKA PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    x_positions,
    cka_values,
    marker="o"
)


plt.axhline(
    y=TRIPLET_ONLY_E5_CKA,
    linestyle="--",
    label="Triplet-only E5 CKA"
)


plt.xticks(
    x_positions,
    checkpoint_names
)


plt.xlabel(
    "Checkpoint"
)


plt.ylabel(
    "Linear CKA"
)


plt.title(
    "Phase 2D.6 Private-Utility Representation Similarity"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    CKA_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 25. CROSS-CORRELATION PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    x_positions,
    mean_correlations,
    marker="o",
    label="Mean Absolute Cross-Correlation"
)


plt.plot(
    x_positions,
    rms_correlations,
    marker="o",
    label="RMS Cross-Correlation"
)


plt.axhline(
    y=TRIPLET_ONLY_E5_MEAN_ABS_CORR,
    linestyle="--",
    label="Triplet-only E5 Mean |r|"
)


plt.xticks(
    x_positions,
    checkpoint_names
)


plt.xlabel(
    "Checkpoint"
)


plt.ylabel(
    "Cross-Correlation"
)


plt.title(
    "Phase 2D.6 Private-Utility Cross-Correlation"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    CORRELATION_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 26. IDENTITY LEAKAGE VS CKA
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    cka_values,
    utility_probe_accuracies,
    s=80
)


for result in results:

    plt.annotate(
        result[
            "checkpoint"
        ],

        (
            result[
                "linear_cka"
            ],

            result[
                "utility_probe_accuracy"
            ]
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points"
    )


plt.xlabel(
    "Private-Utility Linear CKA"
)


plt.ylabel(
    "Utility PID Probe Accuracy (%)"
)


plt.title(
    "Phase 2D.6 Dependence vs Identity Leakage"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    LEAKAGE_CKA_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 27. IDENTITY LEAKAGE VS MEAN CORRELATION
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    mean_correlations,
    utility_probe_accuracies,
    s=80
)


for result in results:

    plt.annotate(
        result[
            "checkpoint"
        ],

        (
            result[
                "mean_absolute_cross_correlation"
            ],

            result[
                "utility_probe_accuracy"
            ]
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points"
    )


plt.xlabel(
    "Mean Absolute Private-Utility Cross-Correlation"
)


plt.ylabel(
    "Utility PID Probe Accuracy (%)"
)


plt.title(
    "Cross-Correlation vs Identity Leakage"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    LEAKAGE_CORRELATION_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 28. DEPENDENCE / IDENTITY-LEAKAGE ASSOCIATION
# ============================================================

cka_array = np.array(
    cka_values
)


correlation_array = np.array(
    mean_correlations
)


leakage_array = np.array(
    utility_probe_accuracies
)


cka_leakage_correlation = (
    np.corrcoef(
        cka_array,
        leakage_array
    )[0, 1]
)


mean_corr_leakage_correlation = (
    np.corrcoef(
        correlation_array,
        leakage_array
    )[0, 1]
)


print()

print("=" * 80)

print(
    "DEPENDENCE / IDENTITY-LEAKAGE ASSOCIATION"
)

print("=" * 80)

print()


print(
    "Pearson correlation between "
    "Linear CKA and utility PID leakage:"
)


print(
    f"{cka_leakage_correlation:.6f}"
)


print()


print(
    "Pearson correlation between "
    "mean absolute cross-correlation "
    "and utility PID leakage:"
)


print(
    f"{mean_corr_leakage_correlation:.6f}"
)


# ============================================================
# 29. INTERPRETATION
# ============================================================

print()

print("=" * 80)

print(
    "INTERPRETATION"
)

print("=" * 80)

print()


if (
    final_result[
        "linear_cka"
    ]
    <
    TRIPLET_ONLY_E5_CKA
):

    print(
        "Phase 2D.6 Epoch 5 has lower private-utility "
        "Linear CKA than the triplet-only Epoch 5 model."
    )

else:

    print(
        "Phase 2D.6 Epoch 5 does not reduce Linear CKA "
        "relative to the triplet-only Epoch 5 model."
    )


if (
    final_result[
        "mean_absolute_cross_correlation"
    ]
    <
    TRIPLET_ONLY_E5_MEAN_ABS_CORR
):

    print(
        "Phase 2D.6 Epoch 5 also has lower mean absolute "
        "private-utility cross-correlation than the "
        "triplet-only Epoch 5 model."
    )

else:

    print(
        "Phase 2D.6 Epoch 5 does not reduce mean absolute "
        "cross-correlation relative to triplet-only Epoch 5."
    )


print()


print(
    f"Utility leakage changed from "
    f"{TRIPLET_ONLY_E5_PID_LEAKAGE:.2f}% "
    f"in triplet-only Epoch 5 to "
    f"{final_result['utility_probe_accuracy']:.2f}% "
    f"in Triplet+Decorrelation Epoch 5."
)


print()


print(
    "These results should be interpreted together with the "
    "independent cross-camera utility results."
)


print(
    "For Phase 2D.6 Epoch 5, the cross-camera analysis "
    "already showed strong same-ID/different-camera "
    "separation."
)


# ============================================================
# 30. FINISH
# ============================================================

print()

print("=" * 80)


print(
    "Phase 2D.6 private-utility dependence analysis completed."
)


print("=" * 80)

print()


print(
    "Saved:"
)


print(
    CSV_PATH
)


print(
    CKA_PLOT_PATH
)


print(
    CORRELATION_PLOT_PATH
)


print(
    LEAKAGE_CKA_PLOT_PATH
)


print(
    LEAKAGE_CORRELATION_PLOT_PATH
)