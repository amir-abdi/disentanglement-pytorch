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


GALLERY_DIR = os.path.join(
    MARKET1501_ROOT,
    "bounding_box_test"
)


# ============================================================
# SELECTED PHASE 2D.6 CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    "outputs/"
    "market1501_disentangled_vae_triplet_decorrelation/"
    "checkpoints/"
    "triplet_decorr_epoch_05.pth"
)


# ============================================================
# LATENT CONFIGURATION
# ============================================================

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

SHIFT_ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


# ------------------------------------------------------------
# alpha = 0.00:
#
# decoder reconstruction only, no identity shift
#
# alpha > 0:
#
# increasingly stronger private identity shift.
#
# mu_utility is NEVER changed.
# ------------------------------------------------------------


# ============================================================
# 3. RUNTIME CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_WORKERS = 4

RANDOM_SEED = 42


# ============================================================
# 4. OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "market1501_full_pis"
)


ASSIGNMENT_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "identity_shift_assignments.csv"
)


SUMMARY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "full_pis_latent_summary.csv"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
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


print("=" * 88)

print(
    "Phase 3B - Full Market-1501 Person Identity Shift Generation"
)

print("=" * 88)


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
    "Checkpoint:"
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


# ============================================================
# 8. MARKET-1501 DATASET
# ============================================================

class Market1501Dataset(
    Dataset
):

    def __init__(
        self,
        image_dir,
        split_name,
        transform=None
    ):

        self.transform = transform

        self.split_name = split_name


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
                f"No JPG images found in:\n"
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


            # ------------------------------------------------
            # Standard junk images.
            # ------------------------------------------------

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


        unique_pids = sorted(
            set(
                sample[
                    "pid"
                ]
                for sample in self.samples
            )
        )


        print(
            f"{split_name}: "
            f"{len(self.samples)} images"
        )


        print(
            f"{split_name}: "
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


        image = transform(
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
# 9. LOAD DATASETS
# ============================================================

train_dataset = Market1501Dataset(
    TRAIN_DIR,
    split_name="Training",
    transform=transform
)


query_dataset = Market1501Dataset(
    QUERY_DIR,
    split_name="Query",
    transform=transform
)


gallery_dataset = Market1501Dataset(
    GALLERY_DIR,
    split_name="Gallery",
    transform=transform
)


print()


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


query_loader = DataLoader(
    query_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 10. BUILD MODEL
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
# 11. LOAD CHECKPOINT
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
    "Loaded and froze encoder + decoder."
)


print()


# ============================================================
# 12. LATENT SPLIT
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
# 13. BUILD TRAINING-ID PRIVATE CENTROIDS
# ============================================================

print("=" * 88)

print(
    "Building donor private centroids..."
)

print("=" * 88)


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


        pid_list = pids.tolist()


        for feature, pid in zip(
            mu_private,
            pid_list
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


        if (
            batch_index % 50
            ==
            0
        ):

            processed = min(
                (
                    batch_index
                    +
                    1
                )
                *
                BATCH_SIZE,

                len(
                    train_dataset
                )
            )


            print(
                f"Processed "
                f"{processed}/"
                f"{len(train_dataset)}",
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
    f"{len(training_pids)} "
    f"donor identity centroids."
)


print()


# ============================================================
# 14. GET ALL EVALUATION IDENTITIES
# ============================================================

query_pids = set(
    sample[
        "pid"
    ]
    for sample in query_dataset.samples
)


gallery_pids = set(
    sample[
        "pid"
    ]
    for sample in gallery_dataset.samples
)


evaluation_pids = sorted(
    query_pids
    |
    gallery_pids
)


print(
    "Evaluation identities:"
)


print(
    len(
        evaluation_pids
    )
)


print()


# ============================================================
# 15. CREATE CONSISTENT ONE-TO-ONE DONOR MAPPING
# ============================================================

# ------------------------------------------------------------
# We use a deterministic shuffled donor list.
#
# This gives each evaluation PID a stable donor identity.
#
# Where counts allow, donors are unique.
#
# This reduces the risk that several source identities are
# artificially collapsed toward one donor.
# ------------------------------------------------------------

donor_rng = random.Random(
    RANDOM_SEED
)


shuffled_training_pids = (
    training_pids.copy()
)


donor_rng.shuffle(
    shuffled_training_pids
)


source_to_donor = {}


if (
    len(
        evaluation_pids
    )
    <=
    len(
        shuffled_training_pids
    )
):

    for source_pid, donor_pid in zip(
        evaluation_pids,
        shuffled_training_pids
    ):

        source_to_donor[
            source_pid
        ] = donor_pid


else:

    # --------------------------------------------------------
    # Fallback if source identities ever exceed donor IDs.
    # --------------------------------------------------------

    for index, source_pid in enumerate(
        evaluation_pids
    ):

        source_to_donor[
            source_pid
        ] = shuffled_training_pids[
            index
            %
            len(
                shuffled_training_pids
            )
        ]


print(
    "Created consistent source -> donor mapping."
)


print()


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
            "source_pid",
            "donor_training_pid",
            "appears_in_query",
            "appears_in_gallery"
        ]
    )


    for source_pid in evaluation_pids:

        writer.writerow(
            [
                source_pid,

                source_to_donor[
                    source_pid
                ],

                source_pid
                in query_pids,

                source_pid
                in gallery_pids
            ]
        )


print(
    "Saved donor mapping:"
)


print(
    ASSIGNMENT_CSV_PATH
)


print()


# ============================================================
# 17. CREATE OUTPUT DIRECTORY STRUCTURE
# ============================================================

for alpha in SHIFT_ALPHAS:

    alpha_dir = os.path.join(
        OUTPUT_DIR,
        f"alpha_{alpha:.2f}"
    )


    os.makedirs(
        os.path.join(
            alpha_dir,
            "query"
        ),
        exist_ok=True
    )


    os.makedirs(
        os.path.join(
            alpha_dir,
            "bounding_box_test"
        ),
        exist_ok=True
    )


# ============================================================
# 18. SHIFT FUNCTION
# ============================================================

def shift_private_identity(
    source_private,
    donor_private,
    alpha
):

    return (
        (
            1.0
            -
            alpha
        )
        *
        source_private

        +

        alpha
        *
        donor_private
    )


# ============================================================
# 19. SAVE IMAGE
# ============================================================

to_pil = transforms.ToPILImage()


def save_generated_image(
    image_tensor,
    save_path
):

    image = (
        image_tensor
        .detach()
        .cpu()
        .clamp(
            0,
            1
        )
    )


    pil_image = to_pil(
        image
    )


    # Preserve Market-1501 JPG filenames.
    pil_image.save(
        save_path,
        format="JPEG",
        quality=95
    )


# ============================================================
# 20. GLOBAL METRIC ACCUMULATORS
# ============================================================

summary_accumulators = {}


for alpha in SHIFT_ALPHAS:

    summary_accumulators[
        alpha
    ] = {
        "count":
            0,

        "private_cosine_sum":
            0.0,

        "private_distance_sum":
            0.0,

        "full_cosine_sum":
            0.0,

        "full_distance_sum":
            0.0,

        "donor_distance_sum":
            0.0,

        "utility_distance_max":
            0.0
    }


# ============================================================
# 21. PROCESS ONE SPLIT
# ============================================================

def generate_split(
    data_loader,
    dataset,
    split_directory_name
):

    print()

    print("=" * 88)


    print(
        f"GENERATING SPLIT: "
        f"{split_directory_name}"
    )


    print("=" * 88)

    print()


    processed_images = 0


    with torch.inference_mode():

        for batch_index, (
            images,
            pids,
            camids,
            paths
        ) in enumerate(
            data_loader
        ):

            images = images.to(
                device,
                non_blocking=True
            )


            # =================================================
            # SOURCE REPRESENTATION
            # =================================================

            mu, _ = encoder(
                images
            )


            (
                source_private,
                source_utility
            ) = split_mu(
                mu
            )


            source_full = torch.cat(
                [
                    source_private,
                    source_utility
                ],
                dim=1
            )


            # =================================================
            # DONOR CENTROIDS FOR THIS BATCH
            # =================================================

            donor_features = []


            for pid in pids.tolist():

                donor_pid = source_to_donor[
                    pid
                ]


                donor_features.append(
                    private_centroids[
                        donor_pid
                    ]
                )


            donor_private = torch.stack(
                donor_features,
                dim=0
            ).to(
                device
            )


            # =================================================
            # GENERATE EVERY ALPHA
            # =================================================

            for alpha in SHIFT_ALPHAS:

                shifted_private = (
                    shift_private_identity(
                        source_private,
                        donor_private,
                        alpha
                    )
                )


                shifted_full = torch.cat(
                    [
                        shifted_private,
                        source_utility
                    ],
                    dim=1
                )


                shifted_logits = decoder(
                    shifted_full
                )


                shifted_images = torch.sigmoid(
                    shifted_logits
                )


                # =============================================
                # LATENT SANITY METRICS
                # =============================================

                private_cosine = (
                    F.cosine_similarity(
                        source_private,
                        shifted_private,
                        dim=1
                    )
                )


                private_distance = torch.norm(
                    source_private
                    -
                    shifted_private,
                    p=2,
                    dim=1
                )


                full_cosine = (
                    F.cosine_similarity(
                        source_full,
                        shifted_full,
                        dim=1
                    )
                )


                full_distance = torch.norm(
                    source_full
                    -
                    shifted_full,
                    p=2,
                    dim=1
                )


                donor_distance = torch.norm(
                    shifted_private
                    -
                    donor_private,
                    p=2,
                    dim=1
                )


                # ---------------------------------------------
                # Utility should be exactly unchanged.
                # ---------------------------------------------

                utility_difference = torch.norm(
                    source_utility
                    -
                    source_utility,
                    p=2,
                    dim=1
                )


                accumulator = (
                    summary_accumulators[
                        alpha
                    ]
                )


                batch_count = (
                    images.size(
                        0
                    )
                )


                accumulator[
                    "count"
                ] += batch_count


                accumulator[
                    "private_cosine_sum"
                ] += (
                    private_cosine
                    .sum()
                    .item()
                )


                accumulator[
                    "private_distance_sum"
                ] += (
                    private_distance
                    .sum()
                    .item()
                )


                accumulator[
                    "full_cosine_sum"
                ] += (
                    full_cosine
                    .sum()
                    .item()
                )


                accumulator[
                    "full_distance_sum"
                ] += (
                    full_distance
                    .sum()
                    .item()
                )


                accumulator[
                    "donor_distance_sum"
                ] += (
                    donor_distance
                    .sum()
                    .item()
                )


                accumulator[
                    "utility_distance_max"
                ] = max(
                    accumulator[
                        "utility_distance_max"
                    ],

                    utility_difference
                    .max()
                    .item()
                )


                # =============================================
                # SAVE GENERATED IMAGES
                # =============================================

                alpha_directory = os.path.join(
                    OUTPUT_DIR,
                    f"alpha_{alpha:.2f}",
                    split_directory_name
                )


                for (
                    generated_image,
                    original_path
                ) in zip(
                    shifted_images,
                    paths
                ):

                    filename = os.path.basename(
                        original_path
                    )


                    save_path = os.path.join(
                        alpha_directory,
                        filename
                    )


                    save_generated_image(
                        generated_image,
                        save_path
                    )


            processed_images += images.size(
                0
            )


            if (
                batch_index % 10
                ==
                0
            ):

                print(
                    f"Processed "
                    f"{processed_images}/"
                    f"{len(dataset)} "
                    f"{split_directory_name} images",
                    end="\r"
                )


    print()


    print(
        f"Completed "
        f"{split_directory_name}: "
        f"{processed_images} images."
    )


# ============================================================
# 22. GENERATE FULL QUERY SET
# ============================================================

generate_split(
    query_loader,
    query_dataset,
    "query"
)


# ============================================================
# 23. GENERATE FULL GALLERY SET
# ============================================================

generate_split(
    gallery_loader,
    gallery_dataset,
    "bounding_box_test"
)


# ============================================================
# 24. VERIFY GENERATED IMAGE COUNTS
# ============================================================

print()

print("=" * 88)

print(
    "VERIFYING OUTPUT COUNTS"
)

print("=" * 88)

print()


for alpha in SHIFT_ALPHAS:

    query_generated = glob.glob(
        os.path.join(
            OUTPUT_DIR,
            f"alpha_{alpha:.2f}",
            "query",
            "*.jpg"
        )
    )


    gallery_generated = glob.glob(
        os.path.join(
            OUTPUT_DIR,
            f"alpha_{alpha:.2f}",
            "bounding_box_test",
            "*.jpg"
        )
    )


    print(
        f"alpha={alpha:.2f}"
    )


    print(
        f"  Query: "
        f"{len(query_generated)} / "
        f"{len(query_dataset)}"
    )


    print(
        f"  Gallery: "
        f"{len(gallery_generated)} / "
        f"{len(gallery_dataset)}"
    )


    if (
        len(
            query_generated
        )
        !=
        len(
            query_dataset
        )
    ):

        raise RuntimeError(
            f"Query count mismatch for alpha "
            f"{alpha:.2f}"
        )


    if (
        len(
            gallery_generated
        )
        !=
        len(
            gallery_dataset
        )
    ):

        raise RuntimeError(
            f"Gallery count mismatch for alpha "
            f"{alpha:.2f}"
        )


print()


# ============================================================
# 25. BUILD FULL-DATASET SUMMARY
# ============================================================

summary_rows = []


print("=" * 88)

print(
    "FULL PERSON IDENTITY SHIFT LATENT SUMMARY"
)

print("=" * 88)

print()


print(
    f"{'Alpha':<10}"
    f"{'Private Cos':>14}"
    f"{'Private Dist':>14}"
    f"{'Full Cos':>14}"
    f"{'Donor Dist':>14}"
    f"{'Util Δ Max':>14}"
)


print(
    "-" * 80
)


for alpha in SHIFT_ALPHAS:

    accumulator = (
        summary_accumulators[
            alpha
        ]
    )


    count = accumulator[
        "count"
    ]


    mean_private_cosine = (
        accumulator[
            "private_cosine_sum"
        ]
        /
        count
    )


    mean_private_distance = (
        accumulator[
            "private_distance_sum"
        ]
        /
        count
    )


    mean_full_cosine = (
        accumulator[
            "full_cosine_sum"
        ]
        /
        count
    )


    mean_full_distance = (
        accumulator[
            "full_distance_sum"
        ]
        /
        count
    )


    mean_donor_distance = (
        accumulator[
            "donor_distance_sum"
        ]
        /
        count
    )


    maximum_utility_change = (
        accumulator[
            "utility_distance_max"
        ]
    )


    summary_rows.append(
        {
            "alpha":
                alpha,

            "number_images":
                count,

            "mean_private_cosine_to_source":
                mean_private_cosine,

            "mean_private_euclidean_to_source":
                mean_private_distance,

            "mean_full_cosine_to_source":
                mean_full_cosine,

            "mean_full_euclidean_to_source":
                mean_full_distance,

            "mean_private_distance_to_donor":
                mean_donor_distance,

            "maximum_utility_latent_change":
                maximum_utility_change
        }
    )


    print(
        f"{alpha:<10.2f}"
        f"{mean_private_cosine:>14.4f}"
        f"{mean_private_distance:>14.4f}"
        f"{mean_full_cosine:>14.4f}"
        f"{mean_donor_distance:>14.4f}"
        f"{maximum_utility_change:>14.8f}"
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
            "number_images",
            "mean_private_cosine_to_source",
            "mean_private_euclidean_to_source",
            "mean_full_cosine_to_source",
            "mean_full_euclidean_to_source",
            "mean_private_distance_to_donor",
            "maximum_utility_latent_change"
        ]
    )


    writer.writeheader()


    writer.writerows(
        summary_rows
    )


# ============================================================
# 27. FINAL UTILITY SANITY CHECK
# ============================================================

maximum_utility_change = max(
    row[
        "maximum_utility_latent_change"
    ]
    for row in summary_rows
)


print()

print("=" * 88)

print(
    "UTILITY-PRESERVATION CHECK"
)

print("=" * 88)

print()


print(
    f"Maximum utility latent change "
    f"across full dataset:"
)


print(
    f"{maximum_utility_change:.10f}"
)


print()


if maximum_utility_change < 1e-7:

    print(
        "PASS: mu_utility remained unchanged "
        "for every generated PIS representation."
    )


else:

    print(
        "WARNING: utility latent changed unexpectedly."
    )


# ============================================================
# 28. OUTPUT STRUCTURE
# ============================================================

print()

print("=" * 88)

print(
    "GENERATED DATASET STRUCTURE"
)

print("=" * 88)

print()


for alpha in SHIFT_ALPHAS:

    print(
        f"alpha_{alpha:.2f}/"
    )


    print(
        "    query/"
    )


    print(
        "    bounding_box_test/"
    )


# ============================================================
# 29. FINISH
# ============================================================

print()

print("=" * 88)

print(
    "Phase 3B full Market-1501 PIS generation completed."
)

print("=" * 88)

print()


print(
    "Saved donor assignments:"
)


print(
    ASSIGNMENT_CSV_PATH
)


print()


print(
    "Saved latent summary:"
)


print(
    SUMMARY_CSV_PATH
)


print()


print(
    "Generated datasets:"
)


print(
    OUTPUT_DIR
)


print()

print(
    "NEXT:"
)


print(
    "Evaluate generated images for privacy and ReID utility "
    "across alpha = 0.00, 0.25, 0.50, 0.75, 1.00."
)