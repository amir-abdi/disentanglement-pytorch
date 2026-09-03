import os
import glob
import re
import csv
import random

import numpy as np

import torch
import torch.nn.functional as F

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

from architectures.decoders.person_conv128x64 import (
    PersonConv128x64
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MARKET1501_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15"
)


TRAIN_DIR = os.path.join(
    MARKET1501_ROOT,
    "bounding_box_train"
)


QUERY_DIR = os.path.join(
    MARKET1501_ROOT,
    "query"
)


# ============================================================
# PHASE 2D.6 SELECTED CHECKPOINT
# ============================================================

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


assert (
    PRIVATE_DIM
    +
    UTILITY_DIM
    ==
    LATENT_DIM
)


# ============================================================
# 2. PIS CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Identity shift strengths
#
# alpha = 0.00
#     no private identity shift
#
# alpha = 0.25
#     weak identity shift
#
# alpha = 0.50
#     medium identity shift
#
# alpha = 0.75
#     strong identity shift
#
# alpha = 1.00
#     private representation replaced completely by
#     donor identity centroid
# ------------------------------------------------------------

SHIFT_ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


# ------------------------------------------------------------
# PILOT SIZE
#
# We first inspect a small number of unseen identities.
#
# After validating this transformation visually and
# numerically, we will apply it to the complete query/gallery
# sets.
# ------------------------------------------------------------

NUM_SOURCE_IDENTITIES = 6

IMAGES_PER_IDENTITY = 3


# ------------------------------------------------------------
# Centroid extraction
# ------------------------------------------------------------

BATCH_SIZE = 64

NUM_WORKERS = 4


RANDOM_SEED = 42


# ============================================================
# 3. OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "person_identity_shift_pilot"
)


GRID_DIR = os.path.join(
    OUTPUT_DIR,
    "grids"
)


IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "shifted_images"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    GRID_DIR,
    exist_ok=True
)


os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)


ASSIGNMENT_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "identity_shift_assignments.csv"
)


METRICS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "identity_shift_metrics.csv"
)


SUMMARY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "identity_shift_summary.csv"
)


# ============================================================
# 4. REPRODUCIBILITY
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
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 84)

print(
    "Phase 3 - Person Identity Shift Pilot"
)

print("=" * 84)


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
    "Selected Phase 2D.6 checkpoint:"
)


print(
    CHECKPOINT_PATH
)


print()


print(
    "Shift strengths:"
)


print(
    SHIFT_ALPHAS
)


print()


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


# ============================================================
# 7. MARKET-1501 DATASET
# ============================================================

class Market1501Dataset(
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


            camid = int(
                match.group(2)
            )


            if pid == -1:

                continue


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


        return (
            image,
            sample[
                "pid"
            ],
            sample[
                "camid"
            ],
            sample[
                "path"
            ]
        )


# ============================================================
# 8. LOAD DATASETS
# ============================================================

train_dataset = Market1501Dataset(
    TRAIN_DIR,
    transform=transform
)


query_dataset = Market1501Dataset(
    QUERY_DIR,
    transform=transform
)


print(
    f"Training images: "
    f"{len(train_dataset)}"
)


print(
    f"Query images: "
    f"{len(query_dataset)}"
)


print(
    f"Training identities: "
    f"{len(set(sample['pid'] for sample in train_dataset.samples))}"
)


print(
    f"Query identities: "
    f"{len(set(sample['pid'] for sample in query_dataset.samples))}"
)


print()


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 9. BUILD ENCODER / DECODER
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


# ============================================================
# 10. LOAD PHASE 2D.6 MODEL
# ============================================================

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


decoder.load_state_dict(
    checkpoint[
        "decoder_state_dict"
    ]
)


encoder.eval()

decoder.eval()


for parameter in encoder.parameters():

    parameter.requires_grad = False


for parameter in decoder.parameters():

    parameter.requires_grad = False


print(
    "Loaded and froze Phase 2D.6 encoder and decoder."
)


print()


# ============================================================
# 11. LATENT HELPER
# ============================================================

def split_mu(
    mu
):

    mu_private = mu[
        :,
        :PRIVATE_DIM
    ]


    mu_utility = mu[
        :,
        PRIVATE_DIM:
    ]


    return (
        mu_private,
        mu_utility
    )


# ============================================================
# 12. BUILD TRAINING-IDENTITY PRIVATE CENTROIDS
# ============================================================

# ------------------------------------------------------------
# A donor identity should represent something more stable
# than one arbitrary image.
#
# We therefore calculate:
#
# c_pid = mean(mu_private)
#
# across all training images belonging to each identity.
# ------------------------------------------------------------

print("=" * 84)

print(
    "Building training identity private centroids..."
)

print("=" * 84)

print()


private_sums = {}

private_counts = {}


with torch.inference_mode():

    for batch_index, (
        images,
        pids,
        camids,
        paths
    ) in enumerate(
        train_loader
    ):

        images = images.to(
            device,
            non_blocking=True
        )


        mu, _ = encoder(
            images
        )


        mu_private, _ = split_mu(
            mu
        )


        mu_private = (
            mu_private
            .detach()
            .cpu()
        )


        pids = pids.tolist()


        for feature, pid in zip(
            mu_private,
            pids
        ):

            if pid not in private_sums:

                private_sums[
                    pid
                ] = torch.zeros_like(
                    feature
                )


                private_counts[
                    pid
                ] = 0


            private_sums[
                pid
            ] += feature


            private_counts[
                pid
            ] += 1


        if batch_index % 50 == 0:

            print(
                f"Processed "
                f"{min((batch_index + 1) * BATCH_SIZE, len(train_dataset))}/"
                f"{len(train_dataset)} training images",
                end="\r"
            )


print()


private_centroids = {}


for pid in private_sums:

    private_centroids[
        pid
    ] = (
        private_sums[
            pid
        ]
        /
        private_counts[
            pid
        ]
    )


training_pids = sorted(
    private_centroids.keys()
)


print(
    f"Created "
    f"{len(private_centroids)} "
    f"private identity centroids."
)


print()


# ============================================================
# 13. GROUP QUERY IMAGES BY PID
# ============================================================

query_pid_to_samples = {}


for sample in query_dataset.samples:

    pid = sample[
        "pid"
    ]


    if pid not in query_pid_to_samples:

        query_pid_to_samples[
            pid
        ] = []


    query_pid_to_samples[
        pid
    ].append(
        sample
    )


# ============================================================
# 14. FIND SUITABLE PILOT IDENTITIES
# ============================================================

# ------------------------------------------------------------
# Prefer identities appearing across at least two cameras.
# ------------------------------------------------------------

candidate_source_pids = []


for pid, samples in query_pid_to_samples.items():

    cameras = set(
        sample[
            "camid"
        ]
        for sample in samples
    )


    if (
        len(cameras) >= 2
        and
        len(samples) >= IMAGES_PER_IDENTITY
    ):

        candidate_source_pids.append(
            pid
        )


pilot_rng = random.Random(
    RANDOM_SEED
)


pilot_rng.shuffle(
    candidate_source_pids
)


selected_source_pids = (
    candidate_source_pids[
        :NUM_SOURCE_IDENTITIES
    ]
)


if len(
    selected_source_pids
) < NUM_SOURCE_IDENTITIES:

    raise RuntimeError(
        "Not enough suitable query identities "
        "for the requested pilot."
    )


print("=" * 84)

print(
    "Selected pilot identities"
)

print("=" * 84)

print()


print(
    selected_source_pids
)


print()


# ============================================================
# 15. CONSISTENT DONOR ASSIGNMENT
# ============================================================

# ------------------------------------------------------------
# VERY IMPORTANT:
#
# Every image belonging to the same source PID receives
# exactly the same donor identity.
#
# This ensures:
#
# Camera 1 source PID  -> donor X
# Camera 2 source PID  -> donor X
# Camera 3 source PID  -> donor X
#
# rather than changing donor independently per image.
# ------------------------------------------------------------

source_to_donor = {}


used_donors = set()


for source_pid in selected_source_pids:

    donor_rng = random.Random(
        RANDOM_SEED
        +
        int(
            source_pid
        )
        *
        1009
    )


    available_donors = [
        pid

        for pid in training_pids

        if (
            pid != source_pid
            and
            pid not in used_donors
        )
    ]


    if len(
        available_donors
    ) == 0:

        available_donors = [
            pid
            for pid in training_pids
            if pid != source_pid
        ]


    donor_pid = donor_rng.choice(
        available_donors
    )


    source_to_donor[
        source_pid
    ] = donor_pid


    used_donors.add(
        donor_pid
    )


# ============================================================
# 16. SAVE DONOR ASSIGNMENTS
# ============================================================

with open(
    ASSIGNMENT_CSV_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )


    writer.writerow(
        [
            "source_query_pid",
            "donor_training_pid"
        ]
    )


    for source_pid in selected_source_pids:

        writer.writerow(
            [
                source_pid,
                source_to_donor[
                    source_pid
                ]
            ]
        )


print(
    "Source -> donor assignments:"
)


for source_pid in selected_source_pids:

    print(
        f"  Query PID "
        f"{source_pid}"
        f" -> "
        f"Training donor PID "
        f"{source_to_donor[source_pid]}"
    )


print()


# ============================================================
# 17. SELECT CROSS-CAMERA EXAMPLES
# ============================================================

def select_cross_camera_samples(
    samples,
    number_samples
):

    # --------------------------------------------------------
    # Prefer a different camera for each selected image.
    # --------------------------------------------------------

    by_camera = {}


    for sample in samples:

        camid = sample[
            "camid"
        ]


        if camid not in by_camera:

            by_camera[
                camid
            ] = []


        by_camera[
            camid
        ].append(
            sample
        )


    selected = []


    cameras = sorted(
        by_camera.keys()
    )


    for camera in cameras:

        if len(
            selected
        ) >= number_samples:

            break


        camera_rng = random.Random(
            RANDOM_SEED
            +
            camera
            *
            31
        )


        selected.append(
            camera_rng.choice(
                by_camera[
                    camera
                ]
            )
        )


    # --------------------------------------------------------
    # If we still need more, fill from remaining images.
    # --------------------------------------------------------

    if len(
        selected
    ) < number_samples:

        selected_paths = set(
            sample[
                "path"
            ]
            for sample in selected
        )


        remaining = [
            sample

            for sample in samples

            if sample[
                "path"
            ]
            not in selected_paths
        ]


        remaining_rng = random.Random(
            RANDOM_SEED
        )


        remaining_rng.shuffle(
            remaining
        )


        number_needed = (
            number_samples
            -
            len(
                selected
            )
        )


        selected.extend(
            remaining[
                :number_needed
            ]
        )


    return selected


# ============================================================
# 18. LOAD ONE SAMPLE
# ============================================================

def load_image_tensor(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    image = transform(
        image
    )


    return image


# ============================================================
# 19. PIS SHIFT FUNCTION
# ============================================================

def shift_private_identity(
    mu_private,
    donor_centroid,
    alpha
):

    # --------------------------------------------------------
    # Adapted PIS-style latent interpolation:
    #
    # p_shift
    # =
    # (1-alpha) p_source
    # +
    # alpha p_donor
    #
    # Utility representation is handled separately and is
    # never modified here.
    # --------------------------------------------------------

    donor_centroid = donor_centroid.to(
        mu_private.device
    )


    if donor_centroid.dim() == 1:

        donor_centroid = (
            donor_centroid
            .unsqueeze(0)
        )


    donor_centroid = donor_centroid.expand(
        mu_private.size(0),
        -1
    )


    shifted_private = (
        (
            1.0
            -
            alpha
        )
        *
        mu_private

        +

        alpha
        *
        donor_centroid
    )


    return shifted_private


# ============================================================
# 20. LATENT METRICS
# ============================================================

def cosine_similarity_mean(
    features_a,
    features_b
):

    similarities = F.cosine_similarity(
        features_a,
        features_b,
        dim=1
    )


    return similarities


def euclidean_distance(
    features_a,
    features_b
):

    distances = torch.norm(
        features_a
        -
        features_b,
        p=2,
        dim=1
    )


    return distances


# ============================================================
# 21. IMAGE CONVERSION FOR PLOTTING
# ============================================================

def tensor_to_numpy_image(
    tensor
):

    tensor = (
        tensor
        .detach()
        .cpu()
        .clamp(
            0,
            1
        )
    )


    image = tensor.permute(
        1,
        2,
        0
    ).numpy()


    return image


# ============================================================
# 22. RUN PILOT PIS
# ============================================================

metric_rows = []


print("=" * 84)

print(
    "Generating shifted identities..."
)

print("=" * 84)

print()


for source_pid in selected_source_pids:

    donor_pid = source_to_donor[
        source_pid
    ]


    donor_centroid = private_centroids[
        donor_pid
    ].to(
        device
    )


    selected_samples = select_cross_camera_samples(
        query_pid_to_samples[
            source_pid
        ],
        IMAGES_PER_IDENTITY
    )


    source_images = torch.stack(
        [
            load_image_tensor(
                sample[
                    "path"
                ]
            )

            for sample in selected_samples
        ]
    ).to(
        device
    )


    # ========================================================
    # ENCODE SOURCE IMAGES
    # ========================================================

    with torch.inference_mode():

        mu, _ = encoder(
            source_images
        )


    original_private, original_utility = split_mu(
        mu
    )


    original_full = torch.cat(
        [
            original_private,
            original_utility
        ],
        dim=1
    )


    # --------------------------------------------------------
    # Row 0 of grid:
    #
    # actual input images
    # --------------------------------------------------------

    grid_rows = [
        source_images.detach().cpu()
    ]


    grid_labels = [
        "Original"
    ]


    # ========================================================
    # APPLY EACH SHIFT STRENGTH
    # ========================================================

    for alpha in SHIFT_ALPHAS:

        shifted_private = shift_private_identity(
            original_private,
            donor_centroid,
            alpha
        )


        shifted_full = torch.cat(
            [
                shifted_private,
                original_utility
            ],
            dim=1
        )


        # ====================================================
        # DECODE SHIFTED LATENT
        # ====================================================

        with torch.inference_mode():

            shifted_logits = decoder(
                shifted_full
            )


            shifted_images = torch.sigmoid(
                shifted_logits
            )


        grid_rows.append(
            shifted_images.detach().cpu()
        )


        grid_labels.append(
            f"alpha={alpha:.2f}"
        )


        # ====================================================
        # LATENT SANITY METRICS
        # ====================================================

        private_cosine = cosine_similarity_mean(
            original_private,
            shifted_private
        )


        private_distance = euclidean_distance(
            original_private,
            shifted_private
        )


        utility_cosine = cosine_similarity_mean(
            original_utility,
            original_utility
        )


        utility_distance = euclidean_distance(
            original_utility,
            original_utility
        )


        full_cosine = cosine_similarity_mean(
            original_full,
            shifted_full
        )


        full_distance = euclidean_distance(
            original_full,
            shifted_full
        )


        # ----------------------------------------------------
        # Distance from shifted private representation
        # to its selected donor centroid.
        # ----------------------------------------------------

        donor_expanded = (
            donor_centroid
            .unsqueeze(0)
            .expand(
                shifted_private.size(0),
                -1
            )
        )


        donor_distance = euclidean_distance(
            shifted_private,
            donor_expanded
        )


        # ====================================================
        # SAVE INDIVIDUAL SHIFTED IMAGES
        # ====================================================

        alpha_directory = os.path.join(
            IMAGE_DIR,
            f"alpha_{alpha:.2f}",
            f"source_{source_pid}_donor_{donor_pid}"
        )


        os.makedirs(
            alpha_directory,
            exist_ok=True
        )


        for image_index, (
            shifted_image,
            sample
        ) in enumerate(
            zip(
                shifted_images,
                selected_samples
            )
        ):

            original_filename = os.path.basename(
                sample[
                    "path"
                ]
            )


            save_path = os.path.join(
                alpha_directory,
                original_filename
            )


            transforms.ToPILImage()(
                shifted_image
                .detach()
                .cpu()
                .clamp(
                    0,
                    1
                )
            ).save(
                save_path
            )


            metric_rows.append(
                {
                    "source_pid":
                        source_pid,

                    "donor_pid":
                        donor_pid,

                    "camera_id":
                        sample[
                            "camid"
                        ],

                    "filename":
                        original_filename,

                    "alpha":
                        alpha,

                    "private_cosine":
                        private_cosine[
                            image_index
                        ].item(),

                    "private_euclidean":
                        private_distance[
                            image_index
                        ].item(),

                    "utility_cosine":
                        utility_cosine[
                            image_index
                        ].item(),

                    "utility_euclidean":
                        utility_distance[
                            image_index
                        ].item(),

                    "full_cosine":
                        full_cosine[
                            image_index
                        ].item(),

                    "full_euclidean":
                        full_distance[
                            image_index
                        ].item(),

                    "distance_to_donor":
                        donor_distance[
                            image_index
                        ].item()
                }
            )


    # ========================================================
    # 23. SAVE LABELLED GRID
    # ========================================================

    number_rows = len(
        grid_rows
    )


    number_columns = len(
        selected_samples
    )


    figure, axes = plt.subplots(
        number_rows,
        number_columns,
        figsize=(
            3 * number_columns,
            2.8 * number_rows
        )
    )


    if number_columns == 1:

        axes = np.expand_dims(
            axes,
            axis=1
        )


    for row_index in range(
        number_rows
    ):

        row_images = grid_rows[
            row_index
        ]


        for column_index in range(
            number_columns
        ):

            axis = axes[
                row_index,
                column_index
            ]


            axis.imshow(
                tensor_to_numpy_image(
                    row_images[
                        column_index
                    ]
                )
            )


            axis.axis(
                "off"
            )


            if row_index == 0:

                camera_id = selected_samples[
                    column_index
                ][
                    "camid"
                ]


                axis.set_title(
                    f"Camera {camera_id}"
                )


            if column_index == 0:

                axis.text(
                    -0.08,
                    0.5,
                    grid_labels[
                        row_index
                    ],
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10
                )


    figure.suptitle(
        (
            f"Source PID {source_pid} "
            f"→ Donor PID {donor_pid}"
        ),
        fontsize=14
    )


    plt.tight_layout()


    grid_path = os.path.join(
        GRID_DIR,
        (
            f"source_{source_pid}_"
            f"donor_{donor_pid}.png"
        )
    )


    plt.savefig(
        grid_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close(
        figure
    )


    print(
        f"Generated PID "
        f"{source_pid} "
        f"-> donor "
        f"{donor_pid}"
    )


    print(
        f"Saved grid:"
    )


    print(
        grid_path
    )


    print()


# ============================================================
# 24. SAVE SAMPLE-LEVEL METRICS
# ============================================================

metric_fieldnames = [
    "source_pid",
    "donor_pid",
    "camera_id",
    "filename",
    "alpha",
    "private_cosine",
    "private_euclidean",
    "utility_cosine",
    "utility_euclidean",
    "full_cosine",
    "full_euclidean",
    "distance_to_donor"
]


with open(
    METRICS_CSV_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=metric_fieldnames
    )


    writer.writeheader()


    writer.writerows(
        metric_rows
    )


# ============================================================
# 25. SUMMARY BY SHIFT STRENGTH
# ============================================================

summary_rows = []


print()

print("=" * 84)

print(
    "PERSON IDENTITY SHIFT SUMMARY"
)

print("=" * 84)

print()


print(
    f"{'Alpha':<10}"
    f"{'Private Cos':>14}"
    f"{'Private Dist':>14}"
    f"{'Full Cos':>14}"
    f"{'Donor Dist':>14}"
)


print(
    "-" * 66
)


for alpha in SHIFT_ALPHAS:

    alpha_rows = [
        row

        for row in metric_rows

        if abs(
            row[
                "alpha"
            ]
            -
            alpha
        ) < 1e-8
    ]


    mean_private_cosine = float(
        np.mean(
            [
                row[
                    "private_cosine"
                ]
                for row in alpha_rows
            ]
        )
    )


    mean_private_distance = float(
        np.mean(
            [
                row[
                    "private_euclidean"
                ]
                for row in alpha_rows
            ]
        )
    )


    mean_utility_cosine = float(
        np.mean(
            [
                row[
                    "utility_cosine"
                ]
                for row in alpha_rows
            ]
        )
    )


    mean_utility_distance = float(
        np.mean(
            [
                row[
                    "utility_euclidean"
                ]
                for row in alpha_rows
            ]
        )
    )


    mean_full_cosine = float(
        np.mean(
            [
                row[
                    "full_cosine"
                ]
                for row in alpha_rows
            ]
        )
    )


    mean_full_distance = float(
        np.mean(
            [
                row[
                    "full_euclidean"
                ]
                for row in alpha_rows
            ]
        )
    )


    mean_donor_distance = float(
        np.mean(
            [
                row[
                    "distance_to_donor"
                ]
                for row in alpha_rows
            ]
        )
    )


    summary_rows.append(
        {
            "alpha":
                alpha,

            "mean_private_cosine":
                mean_private_cosine,

            "mean_private_euclidean":
                mean_private_distance,

            "mean_utility_cosine":
                mean_utility_cosine,

            "mean_utility_euclidean":
                mean_utility_distance,

            "mean_full_cosine":
                mean_full_cosine,

            "mean_full_euclidean":
                mean_full_distance,

            "mean_distance_to_donor":
                mean_donor_distance
        }
    )


    print(
        f"{alpha:<10.2f}"
        f"{mean_private_cosine:>14.4f}"
        f"{mean_private_distance:>14.4f}"
        f"{mean_full_cosine:>14.4f}"
        f"{mean_donor_distance:>14.4f}"
    )


# ============================================================
# 26. SAVE SUMMARY CSV
# ============================================================

with open(
    SUMMARY_CSV_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "alpha",
            "mean_private_cosine",
            "mean_private_euclidean",
            "mean_utility_cosine",
            "mean_utility_euclidean",
            "mean_full_cosine",
            "mean_full_euclidean",
            "mean_distance_to_donor"
        ]
    )


    writer.writeheader()


    writer.writerows(
        summary_rows
    )


# ============================================================
# 27. SANITY CHECK
# ============================================================

maximum_utility_distance = max(
    row[
        "utility_euclidean"
    ]
    for row in metric_rows
)


print()

print("=" * 84)

print(
    "UTILITY-PRESERVATION SANITY CHECK"
)

print("=" * 84)

print()


print(
    "Maximum utility latent change:"
)


print(
    f"{maximum_utility_distance:.10f}"
)


print()


if maximum_utility_distance < 1e-7:

    print(
        "PASS: mu_utility remained unchanged during "
        "the identity shift."
    )


else:

    print(
        "WARNING: mu_utility changed unexpectedly."
    )


# ============================================================
# 28. FINISH
# ============================================================

print()

print("=" * 84)

print(
    "Phase 3 Person Identity Shift pilot completed."
)

print("=" * 84)

print()


print(
    "Saved donor assignments:"
)


print(
    ASSIGNMENT_CSV_PATH
)


print()


print(
    "Saved shift metrics:"
)


print(
    METRICS_CSV_PATH
)


print()


print(
    "Saved shift summary:"
)


print(
    SUMMARY_CSV_PATH
)


print()


print(
    "Saved qualitative grids:"
)


print(
    GRID_DIR
)


print()


print(
    "NEXT:"
)


print(
    "Inspect the generated identity-shift grids before "
    "applying PIS to the complete Market-1501 "
    "query/gallery sets."
)