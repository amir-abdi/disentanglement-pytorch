import os
import glob
import re
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from scipy.stats import ttest_ind

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


CHECKPOINT_PATH = (
    "outputs/"
    "market1501_disentangled_vae_triplet_decorrelation/"
    "checkpoints/"
    "triplet_decorr_epoch_05.pth"
)


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64

NUM_CHANNELS = 3

LATENT_DIM = 128

PRIVATE_DIM = 64
UTILITY_DIM = 64

NUM_PAIRS = 1000

RANDOM_SEED = 42


random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 76)

print(
    "Phase 2D Utility Cross-Camera Analysis"
)

print("=" * 76)


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
# 3. PREPROCESSING
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


# ============================================================
# 4. LOAD MARKET-1501 METADATA
# ============================================================

pattern = re.compile(
    r"([-\d]+)_c(\d)"
)


image_paths = sorted(
    glob.glob(
        os.path.join(
            MARKET1501_TRAIN_DIR,
            "*.jpg"
        )
    )
)


samples = []


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


    samples.append(
        {
            "path":
                image_path,

            "pid":
                pid,

            "camid":
                camid
        }
    )


print(
    f"Loaded {len(samples)} images"
)


unique_pids = sorted(
    set(
        sample["pid"]
        for sample in samples
    )
)


print(
    f"Loaded {len(unique_pids)} identities"
)

print()


# ============================================================
# 5. GROUP IMAGES BY PERSON
# ============================================================

pid_to_samples = {}


for sample in samples:

    pid = sample[
        "pid"
    ]


    if pid not in pid_to_samples:

        pid_to_samples[
            pid
        ] = []


    pid_to_samples[
        pid
    ].append(
        sample
    )


# Keep identities that appear in more than one camera.

cross_camera_pids = []


for pid, pid_samples in pid_to_samples.items():

    cameras = set(
        sample["camid"]
        for sample in pid_samples
    )


    if len(cameras) >= 2:

        cross_camera_pids.append(
            pid
        )


print(
    f"Identities with multiple cameras: "
    f"{len(cross_camera_pids)}"
)

print()


# ============================================================
# 6. BUILD ENCODER
# ============================================================

encoder = PersonGaussianConv128x64(
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS,
    image_size=None
).to(
    device
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)


encoder.load_state_dict(
    checkpoint[
        "encoder_state_dict"
    ]
)


encoder.eval()


for parameter in encoder.parameters():

    parameter.requires_grad = False


print(
    "Loaded and froze encoder."
)

print()


# ============================================================
# 7. FEATURE EXTRACTION
# ============================================================

feature_cache = {}


def extract_latents(
    image_path
):

    if image_path in feature_cache:

        return feature_cache[
            image_path
        ]


    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    image = transform(
        image
    ).unsqueeze(
        0
    ).to(
        device
    )


    with torch.no_grad():

        mu, _ = encoder(
            image
        )


    mu_private = mu[
        :,
        :PRIVATE_DIM
    ]


    mu_utility = mu[
        :,
        PRIVATE_DIM:
    ]


    mu_private = (
        mu_private
        .squeeze(0)
        .cpu()
    )


    mu_utility = (
        mu_utility
        .squeeze(0)
        .cpu()
    )


    feature_cache[
        image_path
    ] = (
        mu_private,
        mu_utility
    )


    return (
        mu_private,
        mu_utility
    )


# ============================================================
# 8. DISTANCE FUNCTIONS
# ============================================================

def euclidean_distance(
    feature_a,
    feature_b
):

    return torch.norm(
        feature_a
        -
        feature_b,
        p=2
    ).item()


def cosine_similarity(
    feature_a,
    feature_b
):

    similarity = torch.nn.functional.cosine_similarity(
        feature_a.unsqueeze(0),
        feature_b.unsqueeze(0)
    )


    return similarity.item()


# ============================================================
# 9. BUILD SAME-ID / DIFFERENT-CAMERA PAIRS
# ============================================================

same_pairs = []


attempts = 0


while (
    len(same_pairs)
    <
    NUM_PAIRS
):

    attempts += 1


    pid = random.choice(
        cross_camera_pids
    )


    pid_samples = pid_to_samples[
        pid
    ]


    sample_a = random.choice(
        pid_samples
    )


    different_camera_samples = [
        sample
        for sample in pid_samples
        if sample["camid"]
        != sample_a["camid"]
    ]


    if len(
        different_camera_samples
    ) == 0:

        continue


    sample_b = random.choice(
        different_camera_samples
    )


    same_pairs.append(
        (
            sample_a,
            sample_b
        )
    )


print(
    f"Created {len(same_pairs)} "
    f"same-ID different-camera pairs"
)


# ============================================================
# 10. BUILD DIFFERENT-ID PAIRS
# ============================================================

different_pairs = []


while (
    len(different_pairs)
    <
    NUM_PAIRS
):

    sample_a = random.choice(
        samples
    )


    sample_b = random.choice(
        samples
    )


    if (
        sample_a["pid"]
        ==
        sample_b["pid"]
    ):

        continue


    different_pairs.append(
        (
            sample_a,
            sample_b
        )
    )


print(
    f"Created {len(different_pairs)} "
    f"different-ID pairs"
)

print()


# ============================================================
# 11. ANALYSE PAIRS
# ============================================================

same_private_euclidean = []
same_private_cosine = []

same_utility_euclidean = []
same_utility_cosine = []


different_private_euclidean = []
different_private_cosine = []

different_utility_euclidean = []
different_utility_cosine = []


print(
    "Extracting pairwise distances..."
)


for (
    sample_a,
    sample_b
) in same_pairs:

    private_a, utility_a = extract_latents(
        sample_a["path"]
    )


    private_b, utility_b = extract_latents(
        sample_b["path"]
    )


    same_private_euclidean.append(
        euclidean_distance(
            private_a,
            private_b
        )
    )


    same_private_cosine.append(
        cosine_similarity(
            private_a,
            private_b
        )
    )


    same_utility_euclidean.append(
        euclidean_distance(
            utility_a,
            utility_b
        )
    )


    same_utility_cosine.append(
        cosine_similarity(
            utility_a,
            utility_b
        )
    )


for (
    sample_a,
    sample_b
) in different_pairs:

    private_a, utility_a = extract_latents(
        sample_a["path"]
    )


    private_b, utility_b = extract_latents(
        sample_b["path"]
    )


    different_private_euclidean.append(
        euclidean_distance(
            private_a,
            private_b
        )
    )


    different_private_cosine.append(
        cosine_similarity(
            private_a,
            private_b
        )
    )


    different_utility_euclidean.append(
        euclidean_distance(
            utility_a,
            utility_b
        )
    )


    different_utility_cosine.append(
        cosine_similarity(
            utility_a,
            utility_b
        )
    )


# ============================================================
# 12. SUMMARY FUNCTION
# ============================================================

def print_metric(
    title,
    same_values,
    different_values
):

    same_values = np.array(
        same_values
    )


    different_values = np.array(
        different_values
    )


    statistic, p_value = ttest_ind(
        same_values,
        different_values,
        equal_var=False
    )


    print(
        title
    )


    print(
        f"Same ID / different camera: "
        f"{same_values.mean():.4f} "
        f"± {same_values.std():.4f}"
    )


    print(
        f"Different ID: "
        f"{different_values.mean():.4f} "
        f"± {different_values.std():.4f}"
    )


    print(
        f"Welch t-test p-value: "
        f"{p_value:.6e}"
    )


    print()


# ============================================================
# 13. RESULTS
# ============================================================

print()

print("=" * 76)

print(
    "PRIVATE LATENT RESULTS"
)

print("=" * 76)

print()


print_metric(
    "Private Euclidean Distance",
    same_private_euclidean,
    different_private_euclidean
)


print_metric(
    "Private Cosine Similarity",
    same_private_cosine,
    different_private_cosine
)


print("=" * 76)

print(
    "UTILITY LATENT RESULTS"
)

print("=" * 76)

print()


print_metric(
    "Utility Euclidean Distance",
    same_utility_euclidean,
    different_utility_euclidean
)


print_metric(
    "Utility Cosine Similarity",
    same_utility_cosine,
    different_utility_cosine
)


# ============================================================
# 14. UTILITY HISTOGRAMS
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.hist(
    same_utility_euclidean,
    bins=40,
    alpha=0.6,
    label="Same ID / Different Camera"
)


plt.hist(
    different_utility_euclidean,
    bins=40,
    alpha=0.6,
    label="Different ID"
)


plt.xlabel(
    "Euclidean Distance"
)


plt.ylabel(
    "Frequency"
)


plt.title(
    "mu_utility Cross-Camera Euclidean Distances"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "utility_cross_camera_euclidean.png",
    dpi=150
)


plt.close()


plt.figure(
    figsize=(9, 6)
)


plt.hist(
    same_utility_cosine,
    bins=40,
    alpha=0.6,
    label="Same ID / Different Camera"
)


plt.hist(
    different_utility_cosine,
    bins=40,
    alpha=0.6,
    label="Different ID"
)


plt.xlabel(
    "Cosine Similarity"
)


plt.ylabel(
    "Frequency"
)


plt.title(
    "mu_utility Cross-Camera Cosine Similarity"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "utility_cross_camera_cosine.png",
    dpi=150
)


plt.close()


# ============================================================
# 15. FINISH
# ============================================================

print("=" * 76)

print(
    "Cross-camera utility analysis completed."
)

print("=" * 76)

print()

print(
    "Saved:"
)

print(
    "utility_cross_camera_euclidean.png"
)

print(
    "utility_cross_camera_cosine.png"
)