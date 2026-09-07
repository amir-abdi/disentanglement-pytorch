import os
import glob
import random
import re

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import matplotlib.pyplot as plt

from architectures.encoders.person_conv128x64 import PersonGaussianConv128x64
from architectures.decoders.person_conv128x64 import PersonConv128x64


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

OUTPUT_DIR = "outputs/market1501_vae_128x64/latent_analysis"

INTERPOLATION_DIR = os.path.join(
    OUTPUT_DIR,
    "interpolations"
)

TRAVERSAL_DIR = os.path.join(
    OUTPUT_DIR,
    "traversals"
)

DISTANCE_DIR = os.path.join(
    OUTPUT_DIR,
    "distances"
)

os.makedirs(INTERPOLATION_DIR, exist_ok=True)
os.makedirs(TRAVERSAL_DIR, exist_ok=True)
os.makedirs(DISTANCE_DIR, exist_ok=True)


IMAGE_HEIGHT = 128
IMAGE_WIDTH = 64
NUM_CHANNELS = 3

LATENT_DIM = 128

# Number of interpolation points from Person A -> Person B
NUM_INTERPOLATION_STEPS = 9

# Latent values explored around the original z value
TRAVERSAL_VALUES = [-3, -2, -1, 0, 1, 2, 3]

# Number of latent dimensions to inspect manually
NUM_TRAVERSAL_DIMS = 8

# Number of same/different identity pairs for distance analysis
NUM_DISTANCE_PAIRS = 1000


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("Phase 2C - Market-1501 Latent Space Analysis")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# 3. DATASET
# ============================================================

class Market1501LatentDataset(Dataset):

    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir
        self.transform = transform

        self.image_paths = sorted(
            glob.glob(
                os.path.join(
                    image_dir,
                    "*.jpg"
                )
            )
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No JPG images found in {image_dir}"
            )

        # Parse Market-1501 person IDs from filenames
        self.samples = []

        pattern = re.compile(
            r"([-\d]+)_c(\d)"
        )

        for image_path in self.image_paths:

            match = pattern.search(
                os.path.basename(image_path)
            )

            if match is None:
                continue

            pid = int(match.group(1))
            camid = int(match.group(2))

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
            image = self.transform(image)

        return {
            "image": image,
            "pid": sample["pid"],
            "camid": sample["camid"],
            "path": sample["path"]
        }


transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_HEIGHT, IMAGE_WIDTH)
        ),
        transforms.ToTensor()
    ]
)


dataset = Market1501LatentDataset(
    MARKET1501_TRAIN_DIR,
    transform=transform
)


# ============================================================
# 4. LOAD TRAINED VAE
# ============================================================

encoder = PersonGaussianConv128x64(
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS,
    image_size=None
).to(device)


decoder = PersonConv128x64(
    latent_dim=LATENT_DIM,
    num_channels=NUM_CHANNELS,
    image_size=None
).to(device)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


encoder.load_state_dict(
    checkpoint["encoder_state_dict"]
)

decoder.load_state_dict(
    checkpoint["decoder_state_dict"]
)


encoder.eval()
decoder.eval()


print(
    f"Loaded checkpoint from {CHECKPOINT_PATH}"
)

print()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def encode_image(image):

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        mu, logvar = encoder(image)

    # Use mu as deterministic latent representation.
    #
    # This avoids random variation from sampling and makes
    # comparisons more stable.

    return mu


def decode_latent(z):

    if z.dim() == 1:
        z = z.unsqueeze(0)

    z = z.to(device)

    with torch.no_grad():

        logits = decoder(z)

        image = torch.sigmoid(
            logits
        )

    return image


def get_indices_by_pid():

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

    return pid_to_indices


pid_to_indices = get_indices_by_pid()


# ============================================================
# 6. LATENT INTERPOLATION
# ============================================================

def run_interpolation():

    print(
        "Running latent interpolation..."
    )

    available_pids = list(
        pid_to_indices.keys()
    )

    pid_a, pid_b = random.sample(
        available_pids,
        2
    )

    index_a = random.choice(
        pid_to_indices[pid_a]
    )

    index_b = random.choice(
        pid_to_indices[pid_b]
    )

    sample_a = dataset[index_a]
    sample_b = dataset[index_b]

    image_a = sample_a["image"]
    image_b = sample_b["image"]

    z_a = encode_image(
        image_a
    )

    z_b = encode_image(
        image_b
    )

    alphas = torch.linspace(
        0,
        1,
        NUM_INTERPOLATION_STEPS
    )

    reconstructed_images = []

    for alpha in alphas:

        z_interp = (
            (1 - alpha) * z_a
            +
            alpha * z_b
        )

        reconstructed = decode_latent(
            z_interp
        )

        reconstructed_images.append(
            reconstructed.cpu()
        )

    interpolation_grid = torch.cat(
        reconstructed_images,
        dim=0
    )

    save_path = os.path.join(
        INTERPOLATION_DIR,
        f"pid_{pid_a}_to_pid_{pid_b}.png"
    )

    save_image(
        interpolation_grid,
        save_path,
        nrow=NUM_INTERPOLATION_STEPS
    )

    print(
        f"Interpolation saved to {save_path}"
    )

    print(
        f"Person A PID: {pid_a}"
    )

    print(
        f"Person B PID: {pid_b}"
    )

    print()


# ============================================================
# 7. LATENT TRAVERSAL
# ============================================================

def run_latent_traversal():

    print(
        "Running latent traversal..."
    )

    sample = dataset[
        random.randint(
            0,
            len(dataset) - 1
        )
    ]

    image = sample["image"]

    z_base = encode_image(
        image
    )

    # Choose latent dimensions with largest absolute magnitude.
    #
    # This gives us dimensions that are likely to be more active
    # for this particular image.

    _, active_dims = torch.topk(
        torch.abs(z_base[0]),
        k=NUM_TRAVERSAL_DIMS
    )

    all_rows = []

    for dim in active_dims:

        dim = dim.item()

        row = []

        for value in TRAVERSAL_VALUES:

            z_modified = z_base.clone()

            z_modified[
                0,
                dim
            ] += value

            reconstruction = decode_latent(
                z_modified
            )

            row.append(
                reconstruction.cpu()
            )

        row = torch.cat(
            row,
            dim=0
        )

        all_rows.append(
            row
        )

        print(
            f"Traversed latent dimension {dim}"
        )

    traversal_grid = torch.cat(
        all_rows,
        dim=0
    )

    save_path = os.path.join(
        TRAVERSAL_DIR,
        f"pid_{sample['pid']}_traversal.png"
    )

    save_image(
        traversal_grid,
        save_path,
        nrow=len(TRAVERSAL_VALUES)
    )

    print(
        f"Traversal grid saved to {save_path}"
    )

    print()


# ============================================================
# 8. SAME VS DIFFERENT PERSON DISTANCES
# ============================================================

def run_distance_analysis():

    print(
        "Running same-person vs different-person "
        "latent-distance analysis..."
    )

    valid_same_pids = [
        pid
        for pid, indices
        in pid_to_indices.items()
        if len(indices) >= 2
    ]

    all_pids = list(
        pid_to_indices.keys()
    )

    same_distances = []
    different_distances = []

    # --------------------------------------------------------
    # Same-person pairs
    # --------------------------------------------------------

    for _ in range(
        NUM_DISTANCE_PAIRS
    ):

        pid = random.choice(
            valid_same_pids
        )

        index_1, index_2 = random.sample(
            pid_to_indices[pid],
            2
        )

        image_1 = dataset[
            index_1
        ]["image"]

        image_2 = dataset[
            index_2
        ]["image"]

        z_1 = encode_image(
            image_1
        )

        z_2 = encode_image(
            image_2
        )

        distance = F.pairwise_distance(
            z_1,
            z_2
        ).item()

        same_distances.append(
            distance
        )

    # --------------------------------------------------------
    # Different-person pairs
    # --------------------------------------------------------

    for _ in range(
        NUM_DISTANCE_PAIRS
    ):

        pid_1, pid_2 = random.sample(
            all_pids,
            2
        )

        index_1 = random.choice(
            pid_to_indices[pid_1]
        )

        index_2 = random.choice(
            pid_to_indices[pid_2]
        )

        image_1 = dataset[
            index_1
        ]["image"]

        image_2 = dataset[
            index_2
        ]["image"]

        z_1 = encode_image(
            image_1
        )

        z_2 = encode_image(
            image_2
        )

        distance = F.pairwise_distance(
            z_1,
            z_2
        ).item()

        different_distances.append(
            distance
        )

    same_mean = sum(
        same_distances
    ) / len(
        same_distances
    )

    different_mean = sum(
        different_distances
    ) / len(
        different_distances
    )

    print()
    print(
        "Average same-person latent distance:",
        f"{same_mean:.4f}"
    )

    print(
        "Average different-person latent distance:",
        f"{different_mean:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Save histogram
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        same_distances,
        bins=40,
        alpha=0.6,
        label="Same Person"
    )

    plt.hist(
        different_distances,
        bins=40,
        alpha=0.6,
        label="Different Person"
    )

    plt.xlabel(
        "Euclidean Distance in Latent Space"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Market-1501 VAE Latent Distance Distribution"
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        DISTANCE_DIR,
        "same_vs_different_latent_distance.png"
    )

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.close()

    print(
        f"Distance histogram saved to {save_path}"
    )

    # --------------------------------------------------------
    # Save numerical values
    # --------------------------------------------------------

    results_path = os.path.join(
        DISTANCE_DIR,
        "distance_results.txt"
    )

    with open(
        results_path,
        "w"
    ) as f:

        f.write(
            f"Same-person mean distance: "
            f"{same_mean:.6f}\n"
        )

        f.write(
            f"Different-person mean distance: "
            f"{different_mean:.6f}\n"
        )

    print(
        f"Distance results saved to {results_path}"
    )

    print()


# ============================================================
# 9. RUN ANALYSIS
# ============================================================

run_interpolation()

run_latent_traversal()

run_distance_analysis()


# ============================================================
# 10. FINISH
# ============================================================

print("=" * 70)
print(
    "Phase 2C latent-space analysis completed."
)
print("=" * 70)