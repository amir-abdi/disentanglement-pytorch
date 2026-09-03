import os
import glob
import random
import re
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from architectures.encoders.person_conv128x64 import PersonGaussianConv128x64


# ============================================================
# 1. CONFIGURATION
# ============================================================

MARKET1501_TRAIN_DIR = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15/bounding_box_train"
)

CHECKPOINT_PATH = (
    "outputs/market1501_vae_128x64/checkpoints/vae_epoch_20.pth"
)

OUTPUT_DIR = (
    "outputs/market1501_vae_128x64/"
    "latent_analysis_v2"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64
NUM_CHANNELS = 3
LATENT_DIM = 128

# Number of pairs used for each comparison
NUM_PAIRS = 1000

# For reproducibility
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("Phase 2C - Latent Analysis V2")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# 3. DATASET
# ============================================================

class Market1501LatentDataset(Dataset):

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

            camid = int(
                match.group(2)
            )

            if pid == -1:
                continue

            self.samples.append(
                {
                    "path": image_path,
                    "pid": pid,
                    "camid": camid
                }
            )

        print(
            f"Loaded {len(self.samples)} "
            f"Market-1501 images"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(
            sample["path"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(
                image
            )

        return {
            "image": image,
            "pid": sample["pid"],
            "camid": sample["camid"],
            "path": sample["path"]
        }


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

dataset = Market1501LatentDataset(
    MARKET1501_TRAIN_DIR,
    transform=transform
)


# ============================================================
# 4. BUILD PID / CAMERA INDEX
# ============================================================

pid_to_indices = {}

for index, sample in enumerate(
    dataset.samples
):

    pid = sample["pid"]

    if pid not in pid_to_indices:

        pid_to_indices[pid] = []

    pid_to_indices[pid].append(
        index
    )


# ============================================================
# 5. FIND VALID CROSS-CAMERA IDENTITIES
# ============================================================

cross_camera_pids = []

for pid, indices in pid_to_indices.items():

    cameras = set()

    for index in indices:

        cameras.add(
            dataset.samples[
                index
            ]["camid"]
        )

    if len(cameras) >= 2:

        cross_camera_pids.append(
            pid
        )

print(
    f"Identities appearing in at least "
    f"two cameras: {len(cross_camera_pids)}"
)

print()


# ============================================================
# 6. LOAD TRAINED ENCODER
# ============================================================

encoder = PersonGaussianConv128x64(
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS,
    image_size=None
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

encoder.load_state_dict(
    checkpoint[
        "encoder_state_dict"
    ]
)

encoder.eval()

print(
    f"Loaded encoder from:"
    f"\n{CHECKPOINT_PATH}"
)

print()


# ============================================================
# 7. ENCODING HELPER
# ============================================================

def encode_image(image):

    if image.dim() == 3:

        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        mu, logvar = encoder(
            image
        )

    # Deterministic representation
    return mu


# ============================================================
# 8. DISTANCE METRICS
# ============================================================

def calculate_metrics(
    z1,
    z2
):

    euclidean = F.pairwise_distance(
        z1,
        z2
    ).item()

    cosine = F.cosine_similarity(
        z1,
        z2
    ).item()

    return (
        euclidean,
        cosine
    )


# ============================================================
# 9. SAMPLE SAME-ID CROSS-CAMERA PAIR
# ============================================================

def sample_same_identity_cross_camera():

    pid = random.choice(
        cross_camera_pids
    )

    indices = pid_to_indices[
        pid
    ]

    # Group samples by camera
    camera_to_indices = {}

    for index in indices:

        camid = dataset.samples[
            index
        ]["camid"]

        if camid not in camera_to_indices:

            camera_to_indices[
                camid
            ] = []

        camera_to_indices[
            camid
        ].append(
            index
        )

    cameras = list(
        camera_to_indices.keys()
    )

    cam1, cam2 = random.sample(
        cameras,
        2
    )

    index1 = random.choice(
        camera_to_indices[
            cam1
        ]
    )

    index2 = random.choice(
        camera_to_indices[
            cam2
        ]
    )

    return (
        index1,
        index2
    )


# ============================================================
# 10. SAMPLE DIFFERENT-ID PAIR
# ============================================================

def sample_different_identity_pair():

    pid1, pid2 = random.sample(
        list(
            pid_to_indices.keys()
        ),
        2
    )

    index1 = random.choice(
        pid_to_indices[
            pid1
        ]
    )

    index2 = random.choice(
        pid_to_indices[
            pid2
        ]
    )

    return (
        index1,
        index2
    )


# ============================================================
# 11. RUN ANALYSIS
# ============================================================

same_euclidean = []
same_cosine = []

different_euclidean = []
different_cosine = []


print(
    "Encoding same-person "
    "cross-camera pairs..."
)

for i in range(
    NUM_PAIRS
):

    index1, index2 = (
        sample_same_identity_cross_camera()
    )

    image1 = dataset[
        index1
    ]["image"]

    image2 = dataset[
        index2
    ]["image"]

    z1 = encode_image(
        image1
    )

    z2 = encode_image(
        image2
    )

    euclidean, cosine = (
        calculate_metrics(
            z1,
            z2
        )
    )

    same_euclidean.append(
        euclidean
    )

    same_cosine.append(
        cosine
    )

    if (
        (i + 1) % 100
        == 0
    ):

        print(
            f"Same-ID pairs: "
            f"{i + 1}/{NUM_PAIRS}"
        )


print()

print(
    "Encoding different-person pairs..."
)

for i in range(
    NUM_PAIRS
):

    index1, index2 = (
        sample_different_identity_pair()
    )

    image1 = dataset[
        index1
    ]["image"]

    image2 = dataset[
        index2
    ]["image"]

    z1 = encode_image(
        image1
    )

    z2 = encode_image(
        image2
    )

    euclidean, cosine = (
        calculate_metrics(
            z1,
            z2
        )
    )

    different_euclidean.append(
        euclidean
    )

    different_cosine.append(
        cosine
    )

    if (
        (i + 1) % 100
        == 0
    ):

        print(
            f"Different-ID pairs: "
            f"{i + 1}/{NUM_PAIRS}"
        )


# ============================================================
# 12. SUMMARY STATISTICS
# ============================================================

same_e = torch.tensor(
    same_euclidean
)

diff_e = torch.tensor(
    different_euclidean
)

same_c = torch.tensor(
    same_cosine
)

diff_c = torch.tensor(
    different_cosine
)


same_e_mean = same_e.mean().item()
same_e_std = same_e.std().item()

diff_e_mean = diff_e.mean().item()
diff_e_std = diff_e.std().item()

same_c_mean = same_c.mean().item()
same_c_std = same_c.std().item()

diff_c_mean = diff_c.mean().item()
diff_c_std = diff_c.std().item()


print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print()

print(
    "Same identity / different camera"
)

print(
    f"Euclidean: "
    f"{same_e_mean:.4f} "
    f"± {same_e_std:.4f}"
)

print(
    f"Cosine:    "
    f"{same_c_mean:.4f} "
    f"± {same_c_std:.4f}"
)

print()

print(
    "Different identity"
)

print(
    f"Euclidean: "
    f"{diff_e_mean:.4f} "
    f"± {diff_e_std:.4f}"
)

print(
    f"Cosine:    "
    f"{diff_c_mean:.4f} "
    f"± {diff_c_std:.4f}"
)


# ============================================================
# 13. STATISTICAL TESTS
# ============================================================

euclidean_test = ttest_ind(
    same_euclidean,
    different_euclidean,
    equal_var=False
)

cosine_test = ttest_ind(
    same_cosine,
    different_cosine,
    equal_var=False
)

print()

print(
    "Welch t-test:"
)

print(
    f"Euclidean p-value: "
    f"{euclidean_test.pvalue:.6g}"
)

print(
    f"Cosine p-value: "
    f"{cosine_test.pvalue:.6g}"
)


# ============================================================
# 14. SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "latent_pair_results.csv"
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
            "pair_number",
            "same_euclidean",
            "same_cosine",
            "different_euclidean",
            "different_cosine"
        ]
    )

    for i in range(
        NUM_PAIRS
    ):

        writer.writerow(
            [
                i + 1,
                same_euclidean[i],
                same_cosine[i],
                different_euclidean[i],
                different_cosine[i]
            ]
        )

print()

print(
    f"Saved pair results to:"
    f"\n{csv_path}"
)


# ============================================================
# 15. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "Phase 2C Latent Analysis V2\n\n"
    )

    f.write(
        "Same identity / different camera\n"
    )

    f.write(
        f"Euclidean mean: "
        f"{same_e_mean:.6f}\n"
    )

    f.write(
        f"Euclidean std: "
        f"{same_e_std:.6f}\n"
    )

    f.write(
        f"Cosine mean: "
        f"{same_c_mean:.6f}\n"
    )

    f.write(
        f"Cosine std: "
        f"{same_c_std:.6f}\n\n"
    )

    f.write(
        "Different identity\n"
    )

    f.write(
        f"Euclidean mean: "
        f"{diff_e_mean:.6f}\n"
    )

    f.write(
        f"Euclidean std: "
        f"{diff_e_std:.6f}\n"
    )

    f.write(
        f"Cosine mean: "
        f"{diff_c_mean:.6f}\n"
    )

    f.write(
        f"Cosine std: "
        f"{diff_c_std:.6f}\n\n"
    )

    f.write(
        "Welch t-tests\n"
    )

    f.write(
        f"Euclidean p-value: "
        f"{euclidean_test.pvalue:.10g}\n"
    )

    f.write(
        f"Cosine p-value: "
        f"{cosine_test.pvalue:.10g}\n"
    )


# ============================================================
# 16. EUCLIDEAN HISTOGRAM
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    same_euclidean,
    bins=40,
    alpha=0.6,
    label=(
        "Same Identity "
        "(Cross-Camera)"
    )
)

plt.hist(
    different_euclidean,
    bins=40,
    alpha=0.6,
    label="Different Identity"
)

plt.xlabel(
    "Euclidean Distance"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "VAE Latent Euclidean Distance"
)

plt.legend()

plt.tight_layout()

euclidean_plot_path = os.path.join(
    OUTPUT_DIR,
    "euclidean_distance_distribution.png"
)

plt.savefig(
    euclidean_plot_path,
    dpi=150
)

plt.close()


# ============================================================
# 17. COSINE HISTOGRAM
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    same_cosine,
    bins=40,
    alpha=0.6,
    label=(
        "Same Identity "
        "(Cross-Camera)"
    )
)

plt.hist(
    different_cosine,
    bins=40,
    alpha=0.6,
    label="Different Identity"
)

plt.xlabel(
    "Cosine Similarity"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "VAE Latent Cosine Similarity"
)

plt.legend()

plt.tight_layout()

cosine_plot_path = os.path.join(
    OUTPUT_DIR,
    "cosine_similarity_distribution.png"
)

plt.savefig(
    cosine_plot_path,
    dpi=150
)

plt.close()


# ============================================================
# 18. FINISH
# ============================================================

print()
print(
    f"Saved Euclidean plot to:"
    f"\n{euclidean_plot_path}"
)

print()

print(
    f"Saved cosine plot to:"
    f"\n{cosine_plot_path}"
)

print()

print("=" * 70)
print(
    "Phase 2C latent analysis V2 completed."
)
print("=" * 70)