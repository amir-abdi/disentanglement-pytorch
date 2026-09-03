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


# ============================================================
# 1. CONFIGURATION
# ============================================================

MARKET1501_ROOT = os.path.expanduser(
    "~/Desktop/ResearchResources/Phase1/deep-person-reid/"
    "reid-data/market1501/Market-1501-v15.09.15"
)


QUERY_DIR = os.path.join(
    MARKET1501_ROOT,
    "query"
)


GALLERY_DIR = os.path.join(
    MARKET1501_ROOT,
    "bounding_box_test"
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

RANDOM_SEED = 42


# ============================================================
# 2. CHECKPOINTS
# ============================================================

# Independent utility PID leakage values are included so that
# the final table combines PRIVACY and ReID UTILITY.
#
# These values come from latent_identity_probe.py.

CHECKPOINTS = [
    {
        "name":
            "Baseline Private v2 Epoch 10",

        "short_name":
            "Baseline",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_private_v2/"
                "checkpoints/"
                "private_v2_epoch_10.pth"
            ),

        "utility_pid_leakage":
            5.29
    },

    {
        "name":
            "Triplet+Decorr Epoch 1",

        "short_name":
            "T+D E1",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_01.pth"
            ),

        "utility_pid_leakage":
            16.92
    },

    {
        "name":
            "Triplet+Decorr Epoch 2",

        "short_name":
            "T+D E2",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_02.pth"
            ),

        "utility_pid_leakage":
            17.54
    },

    {
        "name":
            "Triplet+Decorr Epoch 3",

        "short_name":
            "T+D E3",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_03.pth"
            ),

        "utility_pid_leakage":
            18.59
    },

    {
        "name":
            "Triplet+Decorr Epoch 4",

        "short_name":
            "T+D E4",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_04.pth"
            ),

        "utility_pid_leakage":
            17.70
    },

    {
        "name":
            "Triplet+Decorr Epoch 5",

        "short_name":
            "T+D E5",

        "path":
            (
                "outputs/"
                "market1501_disentangled_vae_triplet_decorrelation/"
                "checkpoints/"
                "triplet_decorr_epoch_05.pth"
            ),

        "utility_pid_leakage":
            16.15
    }
]


# ============================================================
# 3. EVALUATION CONFIGURATION
# ============================================================

BATCH_SIZE = 64

NUM_WORKERS = 4

QUERY_CHUNK_SIZE = 128


RANKS_TO_REPORT = [
    1,
    5,
    10,
    20
]


# ------------------------------------------------------------
# Phase 2D.6 metric learning used normalized utility vectors.
#
# We therefore use normalized representations during
# query/gallery retrieval.
# ------------------------------------------------------------

L2_NORMALIZE_FEATURES = True


# ============================================================
# 4. OUTPUT
# ============================================================

OUTPUT_DIR = (
    "outputs/"
    "market1501_latent_reid_phase2d6_comparison"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "phase2d6_latent_reid_results.csv"
)


UTILITY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "phase2d6_utility_tradeoff.csv"
)


RANK1_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "utility_rank1_across_checkpoints.png"
)


MAP_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "utility_map_across_checkpoints.png"
)


TRADEOFF_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "privacy_utility_tradeoff.png"
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
    "Phase 2D.6 Market-1501 Latent ReID "
    "Checkpoint Comparison"
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
    "Query directory:"
)

print(
    QUERY_DIR
)


print()


print(
    "Gallery directory:"
)

print(
    GALLERY_DIR
)


print()


print(
    f"L2 normalization: "
    f"{L2_NORMALIZE_FEATURES}"
)


print()


# ============================================================
# 7. DATASET
# ============================================================

class Market1501EvaluationDataset(
    Dataset
):

    def __init__(
        self,
        image_dir,
        transform=None,
        split_name="unknown"
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
                f"No JPG images found in "
                f"{split_name}:\n"
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


            # Junk images are ignored.
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
                sample["pid"]
                for sample in self.samples
            )
        )


        unique_camids = sorted(
            set(
                sample["camid"]
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


        print(
            f"{split_name}: "
            f"{len(unique_camids)} cameras"
        )


        print()


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
            ]
        )


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


query_dataset = Market1501EvaluationDataset(
    QUERY_DIR,
    transform=transform,
    split_name="Query"
)


gallery_dataset = Market1501EvaluationDataset(
    GALLERY_DIR,
    transform=transform,
    split_name="Gallery"
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
            "'encoder_state_dict'."
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
# 11. FEATURE EXTRACTION
# ============================================================

def extract_features(
    encoder,
    data_loader
):

    private_features = []

    utility_features = []

    full_features = []

    all_pids = []

    all_camids = []


    with torch.inference_mode():

        for (
            images,
            pids,
            camids
        ) in data_loader:

            images = images.to(
                device,
                non_blocking=True
            )


            # ------------------------------------------------
            # Deterministic latent mean.
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


            full_mu = mu


            if L2_NORMALIZE_FEATURES:

                mu_private = F.normalize(
                    mu_private,
                    p=2,
                    dim=1
                )


                mu_utility = F.normalize(
                    mu_utility,
                    p=2,
                    dim=1
                )


                full_mu = F.normalize(
                    full_mu,
                    p=2,
                    dim=1
                )


            private_features.append(
                mu_private.cpu()
            )


            utility_features.append(
                mu_utility.cpu()
            )


            full_features.append(
                full_mu.cpu()
            )


            all_pids.append(
                pids.cpu()
            )


            all_camids.append(
                camids.cpu()
            )


    return {
        "private":
            torch.cat(
                private_features,
                dim=0
            ).float(),

        "utility":
            torch.cat(
                utility_features,
                dim=0
            ).float(),

        "full":
            torch.cat(
                full_features,
                dim=0
            ).float(),

        "pids":
            torch.cat(
                all_pids,
                dim=0
            ).numpy(),

        "camids":
            torch.cat(
                all_camids,
                dim=0
            ).numpy()
    }


# ============================================================
# 12. AVERAGE PRECISION
# ============================================================

def compute_average_precision(
    matches
):

    relevant_indices = np.where(
        matches == 1
    )[0]


    if len(
        relevant_indices
    ) == 0:

        return None


    cumulative_matches = np.cumsum(
        matches
    )


    precisions = []


    for rank_index in relevant_indices:

        precision = (
            cumulative_matches[
                rank_index
            ]
            /
            (
                rank_index
                +
                1
            )
        )


        precisions.append(
            precision
        )


    return float(
        np.mean(
            precisions
        )
    )


# ============================================================
# 13. MARKET-1501 EVALUATION
# ============================================================

def evaluate_reid(
    query_features,
    gallery_features,
    query_pids,
    gallery_pids,
    query_camids,
    gallery_camids,
    representation_name
):

    print(
        f"Evaluating "
        f"{representation_name}..."
    )


    number_queries = (
        query_features.shape[
            0
        ]
    )


    number_gallery = (
        gallery_features.shape[
            0
        ]
    )


    cmc = np.zeros(
        number_gallery,
        dtype=np.float64
    )


    average_precisions = []


    valid_queries = 0


    gallery_device = gallery_features.to(
        device
    )


    for start_index in range(
        0,
        number_queries,
        QUERY_CHUNK_SIZE
    ):

        end_index = min(
            start_index
            +
            QUERY_CHUNK_SIZE,
            number_queries
        )


        query_chunk = query_features[
            start_index:
            end_index
        ].to(
            device
        )


        if L2_NORMALIZE_FEATURES:

            similarity = torch.mm(
                query_chunk,
                gallery_device.T
            )


            distances = (
                2.0
                -
                2.0
                *
                similarity
            )


        else:

            distances = torch.cdist(
                query_chunk,
                gallery_device,
                p=2
            )


        distances = (
            distances
            .cpu()
            .numpy()
        )


        for local_index in range(
            distances.shape[
                0
            ]
        ):

            query_index = (
                start_index
                +
                local_index
            )


            query_pid = query_pids[
                query_index
            ]


            query_camid = query_camids[
                query_index
            ]


            order = np.argsort(
                distances[
                    local_index
                ]
            )


            ordered_pids = gallery_pids[
                order
            ]


            ordered_camids = gallery_camids[
                order
            ]


            # ------------------------------------------------
            # Standard Market-1501 filtering:
            #
            # remove same PID + same camera.
            # ------------------------------------------------

            remove = (
                (
                    ordered_pids
                    ==
                    query_pid
                )
                &
                (
                    ordered_camids
                    ==
                    query_camid
                )
            )


            keep = np.logical_not(
                remove
            )


            ordered_pids = ordered_pids[
                keep
            ]


            matches = (
                ordered_pids
                ==
                query_pid
            ).astype(
                np.int32
            )


            if not np.any(
                matches
            ):

                continue


            valid_queries += 1


            # =================================================
            # CMC
            # =================================================

            first_match_index = (
                np.where(
                    matches == 1
                )[0][0]
            )


            cmc[
                first_match_index:
            ] += 1


            # =================================================
            # mAP
            # =================================================

            average_precision = (
                compute_average_precision(
                    matches
                )
            )


            if average_precision is not None:

                average_precisions.append(
                    average_precision
                )


        print(
            f"  Processed "
            f"{end_index}/"
            f"{number_queries} queries",
            end="\r"
        )


    print()


    if valid_queries == 0:

        raise RuntimeError(
            "No valid queries found."
        )


    cmc = (
        cmc
        /
        valid_queries
    )


    mean_ap = (
        np.mean(
            average_precisions
        )
        *
        100
    )


    result = {
        "representation":
            representation_name,

        "valid_queries":
            valid_queries,

        "mAP":
            float(
                mean_ap
            )
    }


    for rank in RANKS_TO_REPORT:

        result[
            f"Rank-{rank}"
        ] = (
            cmc[
                rank
                -
                1
            ]
            *
            100
        )


    print(
        f"{representation_name} RESULTS"
    )


    print(
        "-" * 50
    )


    print(
        f"mAP: "
        f"{result['mAP']:.2f}%"
    )


    for rank in RANKS_TO_REPORT:

        print(
            f"Rank-{rank}: "
            f"{result[f'Rank-{rank}']:.2f}%"
        )


    print()


    return result


# ============================================================
# 14. EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    checkpoint_info
):

    checkpoint_name = (
        checkpoint_info[
            "name"
        ]
    )


    short_name = (
        checkpoint_info[
            "short_name"
        ]
    )


    checkpoint_path = (
        checkpoint_info[
            "path"
        ]
    )


    utility_pid_leakage = (
        checkpoint_info[
            "utility_pid_leakage"
        ]
    )


    print()

    print("=" * 88)


    print(
        f"EVALUATING CHECKPOINT: "
        f"{checkpoint_name}"
    )


    print("=" * 88)


    print(
        checkpoint_path
    )


    print()


    encoder = load_encoder(
        checkpoint_path
    )


    print(
        "Extracting query features..."
    )


    query_outputs = extract_features(
        encoder,
        query_loader
    )


    print(
        "Extracting gallery features..."
    )


    gallery_outputs = extract_features(
        encoder,
        gallery_loader
    )


    print()


    representations = [
        (
            "mu_private",
            "private"
        ),

        (
            "mu_utility",
            "utility"
        ),

        (
            "full_mu",
            "full"
        )
    ]


    checkpoint_results = []


    for (
        representation_name,
        feature_key
    ) in representations:

        result = evaluate_reid(
            query_outputs[
                feature_key
            ],

            gallery_outputs[
                feature_key
            ],

            query_outputs[
                "pids"
            ],

            gallery_outputs[
                "pids"
            ],

            query_outputs[
                "camids"
            ],

            gallery_outputs[
                "camids"
            ],

            representation_name
        )


        result[
            "checkpoint"
        ] = checkpoint_name


        result[
            "short_checkpoint"
        ] = short_name


        result[
            "utility_pid_leakage"
        ] = utility_pid_leakage


        checkpoint_results.append(
            result
        )


    del encoder

    del query_outputs

    del gallery_outputs


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return checkpoint_results


# ============================================================
# 15. RUN ALL CHECKPOINTS
# ============================================================

all_results = []


for checkpoint_info in CHECKPOINTS:

    checkpoint_results = evaluate_checkpoint(
        checkpoint_info
    )


    all_results.extend(
        checkpoint_results
    )


# ============================================================
# 16. FULL LATENT REID TABLE
# ============================================================

print()

print("=" * 104)

print(
    "FINAL MARKET-1501 PHASE 2D.6 LATENT REID RESULTS"
)

print("=" * 104)

print()


print(
    f"{'Checkpoint':<12}"
    f"{'Representation':<16}"
    f"{'mAP':>10}"
    f"{'Rank-1':>10}"
    f"{'Rank-5':>10}"
    f"{'Rank-10':>10}"
    f"{'Rank-20':>10}"
)


print(
    "-" * 78
)


for result in all_results:

    print(
        f"{result['short_checkpoint']:<12}"
        f"{result['representation']:<16}"
        f"{result['mAP']:>9.2f}%"
        f"{result['Rank-1']:>9.2f}%"
        f"{result['Rank-5']:>9.2f}%"
        f"{result['Rank-10']:>9.2f}%"
        f"{result['Rank-20']:>9.2f}%"
    )


# ============================================================
# 17. SAVE FULL RESULTS CSV
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
            "representation",
            "utility_pid_leakage",
            "valid_queries",
            "mAP",
            "Rank-1",
            "Rank-5",
            "Rank-10",
            "Rank-20"
        ]
    )


    for result in all_results:

        writer.writerow(
            [
                result[
                    "checkpoint"
                ],

                result[
                    "representation"
                ],

                result[
                    "utility_pid_leakage"
                ],

                result[
                    "valid_queries"
                ],

                result[
                    "mAP"
                ],

                result[
                    "Rank-1"
                ],

                result[
                    "Rank-5"
                ],

                result[
                    "Rank-10"
                ],

                result[
                    "Rank-20"
                ]
            ]
        )


# ============================================================
# 18. EXTRACT UTILITY RESULTS
# ============================================================

utility_results = [
    result

    for result in all_results

    if result[
        "representation"
    ]
    ==
    "mu_utility"
]


# ============================================================
# 19. PRIVACY / UTILITY MODEL-SELECTION TABLE
# ============================================================

print()

print("=" * 100)

print(
    "PHASE 2D.6 PRIVACY-UTILITY MODEL SELECTION"
)

print("=" * 100)

print()


print(
    f"{'Checkpoint':<14}"
    f"{'PID Leak':>12}"
    f"{'Utility mAP':>14}"
    f"{'Utility R1':>14}"
    f"{'Utility R5':>14}"
    f"{'Utility R10':>14}"
)


print(
    "-" * 82
)


for result in utility_results:

    print(
        f"{result['short_checkpoint']:<14}"
        f"{result['utility_pid_leakage']:>11.2f}%"
        f"{result['mAP']:>13.2f}%"
        f"{result['Rank-1']:>13.2f}%"
        f"{result['Rank-5']:>13.2f}%"
        f"{result['Rank-10']:>13.2f}%"
    )


# ============================================================
# 20. SAVE UTILITY TRADE-OFF CSV
# ============================================================

with open(
    UTILITY_CSV_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )


    writer.writerow(
        [
            "checkpoint",
            "utility_pid_leakage",
            "utility_mAP",
            "utility_rank1",
            "utility_rank5",
            "utility_rank10",
            "utility_rank20"
        ]
    )


    for result in utility_results:

        writer.writerow(
            [
                result[
                    "short_checkpoint"
                ],

                result[
                    "utility_pid_leakage"
                ],

                result[
                    "mAP"
                ],

                result[
                    "Rank-1"
                ],

                result[
                    "Rank-5"
                ],

                result[
                    "Rank-10"
                ],

                result[
                    "Rank-20"
                ]
            ]
        )


# ============================================================
# 21. UTILITY RANK-1 GRAPH
# ============================================================

checkpoint_labels = [
    result[
        "short_checkpoint"
    ]
    for result in utility_results
]


rank1_values = [
    result[
        "Rank-1"
    ]
    for result in utility_results
]


x_positions = list(
    range(
        len(
            checkpoint_labels
        )
    )
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    x_positions,
    rank1_values,
    marker="o"
)


plt.xticks(
    x_positions,
    checkpoint_labels
)


plt.xlabel(
    "Checkpoint"
)


plt.ylabel(
    "mu_utility Rank-1 (%)"
)


plt.title(
    "Unseen-ID Utility Rank-1 Across "
    "Phase 2D.6 Checkpoints"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    RANK1_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 22. UTILITY mAP GRAPH
# ============================================================

map_values = [
    result[
        "mAP"
    ]
    for result in utility_results
]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    x_positions,
    map_values,
    marker="o"
)


plt.xticks(
    x_positions,
    checkpoint_labels
)


plt.xlabel(
    "Checkpoint"
)


plt.ylabel(
    "mu_utility mAP (%)"
)


plt.title(
    "Unseen-ID Utility mAP Across "
    "Phase 2D.6 Checkpoints"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    MAP_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 23. PRIVACY / UTILITY TRADE-OFF GRAPH
# ============================================================

leakage_values = [
    result[
        "utility_pid_leakage"
    ]
    for result in utility_results
]


plt.figure(
    figsize=(9, 6)
)


plt.scatter(
    leakage_values,
    rank1_values,
    s=90
)


for result in utility_results:

    plt.annotate(
        result[
            "short_checkpoint"
        ],

        (
            result[
                "utility_pid_leakage"
            ],

            result[
                "Rank-1"
            ]
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points"
    )


plt.xlabel(
    "Utility PID Leakage (%)"
)


plt.ylabel(
    "Unseen-ID mu_utility Rank-1 (%)"
)


plt.title(
    "Phase 2D.6 Privacy-Utility Trade-off"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    TRADEOFF_PLOT_PATH,
    dpi=150
)


plt.close()


# ============================================================
# 24. BEST UTILITY CHECKPOINT
# ============================================================

best_rank1_result = max(
    utility_results,
    key=lambda result:
        result[
            "Rank-1"
        ]
)


best_map_result = max(
    utility_results,
    key=lambda result:
        result[
            "mAP"
        ]
)


lowest_leakage_result = min(
    utility_results[
        1:
    ],
    key=lambda result:
        result[
            "utility_pid_leakage"
        ]
)


print()

print("=" * 88)

print(
    "MODEL-SELECTION SUMMARY"
)

print("=" * 88)

print()


print(
    "Highest unseen-ID utility Rank-1:"
)


print(
    f"  Checkpoint: "
    f"{best_rank1_result['short_checkpoint']}"
)


print(
    f"  Rank-1: "
    f"{best_rank1_result['Rank-1']:.2f}%"
)


print(
    f"  mAP: "
    f"{best_rank1_result['mAP']:.2f}%"
)


print(
    f"  PID leakage: "
    f"{best_rank1_result['utility_pid_leakage']:.2f}%"
)


print()


print(
    "Highest unseen-ID utility mAP:"
)


print(
    f"  Checkpoint: "
    f"{best_map_result['short_checkpoint']}"
)


print(
    f"  mAP: "
    f"{best_map_result['mAP']:.2f}%"
)


print(
    f"  Rank-1: "
    f"{best_map_result['Rank-1']:.2f}%"
)


print(
    f"  PID leakage: "
    f"{best_map_result['utility_pid_leakage']:.2f}%"
)


print()


print(
    "Lowest PID leakage among Phase 2D.6 checkpoints:"
)


print(
    f"  Checkpoint: "
    f"{lowest_leakage_result['short_checkpoint']}"
)


print(
    f"  Leakage: "
    f"{lowest_leakage_result['utility_pid_leakage']:.2f}%"
)


print(
    f"  Rank-1: "
    f"{lowest_leakage_result['Rank-1']:.2f}%"
)


print(
    f"  mAP: "
    f"{lowest_leakage_result['mAP']:.2f}%"
)


# ============================================================
# 25. FINISH
# ============================================================

print()

print("=" * 88)

print(
    "Phase 2D.6 Market-1501 checkpoint "
    "evaluation completed."
)

print("=" * 88)

print()


print(
    "Saved:"
)


print(
    CSV_PATH
)


print(
    UTILITY_CSV_PATH
)


print(
    RANK1_PLOT_PATH
)


print(
    MAP_PLOT_PATH
)


print(
    TRADEOFF_PLOT_PATH
)